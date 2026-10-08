"""Structure Exposure Network construction and outcome analysis."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from src.analysis.fragility import FIRES, fit_5pl


M_TO_FT = 3.28084
CUMULATIVE_EDGE_FLOOR_FRACTION = 0.25


def pooled_single_emitter_threshold(data_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Fit the pooled curve and return coupling at absolute P(destroyed)=0.50."""
    analysis = pd.read_parquet(Path(data_dir) / "analysis.parquet")
    exposed = analysis[analysis.exposed.eq(1)].copy()
    fit = fit_5pl(exposed.F_destroyed_wmean, exposed.is_destroyed)
    threshold = coupling_at_probability(fit, .50)
    summary = pd.DataFrame([{
        "population": "Pooled exposed structures",
        "n": len(exposed),
        "p_min": fit["p_min"], "p_max": fit["p_max"],
        "k": fit["k"], "hill": fit["hill"],
        "asymmetry": fit["asymmetry"],
        "fragility_midpoint_F50": fit["f50"],
        "F_ij_threshold": threshold,
        "probability_equivalent": .50,
    }])
    return fit, summary


def coupling_at_probability(fit: dict, probability: float) -> float:
    """Invert a fitted monotone 5PL to an equivalent single-emitter coupling."""
    if not fit["p_min"] < probability < fit["p_max"]:
        raise ValueError("Probability lies outside the fitted asymptotes")
    denominator = (
        ((fit["p_max"] - fit["p_min"]) /
         (probability - fit["p_min"])) ** (1 / fit["asymmetry"]) - 1
    ) ** (1 / fit["hill"])
    return float(fit["k"] / denominator)


def _source_signature(paths: list[Path]) -> dict:
    return {
        path.name: {"size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
    }


def build_within_perimeter_network(data_dir: Path, fire: str,
                                   cache_dir: Path | None = None,
                                   max_distance_m: float = 100.0,
                                   force: bool = False) -> dict:
    """Create one single-neighbour, whole-surface bond per building pair.

    Every mesh building whose representative point falls inside the fire
    perimeter is retained as a node, including buildings with no active bond.
    Edges croseng the perimeter are excluded before components are formed.

    The two directed building-pair contributions each aggregate the complete
    visible surface of one emitter and average over the receiver's sampled
    surface.  Their maximum forms one undirected bond.  Contributions from
    multiple neighbouring buildings are not summed, making the SEN a
    conservative single-emitter or weakest-link construction.
    """
    data_dir = Path(data_dir)
    edge_path = data_dir / "pairs" / f"{fire}_directed_pairs.parquet"
    building_path = data_dir / "nx" / f"{fire}_buildings.parquet"
    perimeter_path = data_dir / "nx" / "fire_perims.parquet"
    sources = [edge_path, building_path, perimeter_path]
    signature = _source_signature(sources)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        node_cache = cache_dir / f"sen_{fire}_nodes.parquet"
        edge_cache = cache_dir / f"sen_{fire}_edges.parquet"
        meta_cache = cache_dir / f"sen_{fire}_network.json"
        if not force and node_cache.exists() and edge_cache.exists() and meta_cache.exists():
            metadata = json.loads(meta_cache.read_text())
            if (metadata.get("sources") == signature and
                    metadata.get("max_distance_m") == max_distance_m):
                nodes = gpd.read_parquet(node_cache)
                edges = pd.read_parquet(edge_cache)
                return {"fire": fire, "nodes": nodes, "edges": edges,
                        "max_distance_m": max_distance_m, "cached": True}

    buildings = gpd.read_parquet(building_path, columns=["BLD_ID", "geometry"])
    buildings["BLD_ID"] = buildings.BLD_ID.astype(str)
    buildings = buildings.drop_duplicates("BLD_ID").reset_index(drop=True)
    perimeters = gpd.read_parquet(perimeter_path).set_index("FIRE_NAME")
    perimeter = gpd.GeoSeries(
        [perimeters.loc[fire].geometry], crs=perimeters.crs
    ).to_crs(buildings.crs).iloc[0]
    inside = buildings.geometry.representative_point().within(perimeter)
    nodes = buildings.loc[inside].reset_index(drop=True)
    nodes["node"] = np.arange(len(nodes), dtype=np.int64)
    node_index = pd.Series(nodes.node.to_numpy(), index=nodes.BLD_ID)

    escaped = str(edge_path).replace("'", "''")
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='3GB'")
    pairs = con.execute(f"""
        WITH visible AS (
            SELECT CAST(source_BLD_ID AS VARCHAR) AS source_id,
                   CAST(target_BLD_ID AS VARCHAR) AS target_id,
                   GREATEST(COALESCE(geom_coupling_wmean, 0), 0) AS coupling,
                   ssd_m
            FROM read_parquet('{escaped}')
            WHERE ssd_m > 0 AND ssd_m <= {float(max_distance_m)}
              AND CAST(source_BLD_ID AS VARCHAR)
                  <> CAST(target_BLD_ID AS VARCHAR)
        )
        SELECT LEAST(source_id, target_id) AS bld_a,
               GREATEST(source_id, target_id) AS bld_b,
               LEAST(MAX(coupling), 1.0) AS F_ij,
               MIN(ssd_m) AS ssd_m
        FROM visible
        WHERE source_id <> 'None' AND target_id <> 'None'
        GROUP BY 1, 2
    """).fetchdf()
    con.close()
    pairs["u"] = pairs.bld_a.map(node_index)
    pairs["v"] = pairs.bld_b.map(node_index)
    edges = pairs.dropna(subset=["u", "v"]).copy()
    edges[["u", "v"]] = edges[["u", "v"]].astype(np.int64)
    edges = edges[["bld_a", "bld_b", "u", "v", "F_ij", "ssd_m"]]

    if cache_dir is not None:
        nodes.to_parquet(node_cache, index=False)
        edges.to_parquet(edge_cache, index=False)
        meta_cache.write_text(json.dumps({
            "fire": fire, "max_distance_m": max_distance_m,
            "sources": signature,
        }, indent=2))
    return {"fire": fire, "nodes": nodes, "edges": edges,
            "max_distance_m": max_distance_m, "cached": False}


def _labels_after_active_edges(n_nodes: int, u: np.ndarray,
                               v: np.ndarray) -> np.ndarray:
    parent = np.arange(n_nodes, dtype=np.int64)
    size = np.ones(n_nodes, dtype=np.int64)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left, right in zip(u, v):
        root_left, root_right = find(int(left)), find(int(right))
        if root_left == root_right:
            continue
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]
    return np.asarray([find(index) for index in range(n_nodes)], dtype=np.int64)


def component_state(network: dict, threshold: float,
                    analysis: pd.DataFrame) -> dict:
    """Form SEN components at a common bond threshold and attach outcomes."""
    nodes = network["nodes"].copy()
    edges = network["edges"]
    active = edges.F_ij.ge(threshold).to_numpy()
    active_edges = edges.loc[active].copy()
    labels = _labels_after_active_edges(
        len(nodes), active_edges.u.to_numpy(), active_edges.v.to_numpy()
    )
    roots, sizes = np.unique(labels, return_counts=True)
    size_lookup = pd.Series(sizes, index=roots)
    nodes["component_id"] = labels
    nodes["component_size"] = size_lookup.loc[labels].to_numpy(int)
    outcomes = (analysis[analysis.fire.eq(network["fire"])]
                .assign(BLD_ID=lambda frame: frame.BLD_ID.astype(str))
                .drop_duplicates("BLD_ID")
                .set_index("BLD_ID")["is_destroyed"])
    nodes["is_destroyed"] = nodes.BLD_ID.map(outcomes)
    nodes["assessed"] = nodes.is_destroyed.notna()

    if len(active_edges):
        active_roots = labels[active_edges.u.to_numpy()]
        component_mean_ssd = (
            pd.Series(active_edges.ssd_m.to_numpy())
            .groupby(active_roots).mean()
        )
        mean_component_ssd_ft = float(component_mean_ssd.mean() * M_TO_FT)
    else:
        component_mean_ssd = pd.Series(dtype=float)
        mean_component_ssd_ft = np.nan
    summary = {
        "fire": network["fire"], "F_ij_threshold": threshold,
        "nodes_inside_perimeter": len(nodes),
        "assessed_inside_perimeter": int(nodes.assessed.sum()),
        "destroyed_inside_perimeter": int(nodes.is_destroyed.fillna(0).sum()),
        "active_bonds": int(active.sum()), "components": len(roots),
        "connected_components": int(len(component_mean_ssd)),
        "connected_buildings": int(nodes.component_size.gt(1).sum()),
        "largest_component": int(sizes.max()),
        "mean_component_ssd_ft": mean_component_ssd_ft,
    }
    return {"fire": network["fire"], "threshold": threshold,
            "nodes": nodes, "active_edges": active_edges,
            "labels": labels, "summary": summary}


def _cumulative_top_n_profile(network: dict, n_neighbors: int) -> dict:
    """Rank incident links once and return fixed top-N node profiles."""
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least one")
    edges = network["edges"]
    n_nodes = len(network["nodes"])
    edge_id = np.arange(len(edges), dtype=np.int64)
    incidence = pd.concat([
        pd.DataFrame({"node": edges.u.to_numpy(np.int64),
                      "edge_id": edge_id,
                      "F_ij": edges.F_ij.to_numpy(float)}),
        pd.DataFrame({"node": edges.v.to_numpy(np.int64),
                      "edge_id": edge_id,
                      "F_ij": edges.F_ij.to_numpy(float)}),
    ], ignore_index=True)
    top_n = (incidence.sort_values(
        ["node", "F_ij", "edge_id"], ascending=[True, False, True],
        kind="stable",
    ).groupby("node", sort=False).head(n_neighbors))
    top_count = np.bincount(
        top_n.node.to_numpy(np.int64), minlength=n_nodes
    ).astype(np.int16)
    top_sum = np.bincount(
        top_n.node.to_numpy(np.int64),
        weights=top_n.F_ij.to_numpy(float), minlength=n_nodes,
    )
    return {
        "top_count": top_count, "top_sum": top_sum,
        "top_node": top_n.node.to_numpy(np.int64),
        "top_edge_id": top_n.edge_id.to_numpy(np.int64),
    }


def _cumulative_top_n_selection(network: dict, threshold: float,
                                n_neighbors: int,
                                profile: dict | None = None,
                                edge_floor_fraction: float =
                                CUMULATIVE_EDGE_FLOOR_FRACTION) -> tuple:
    """Return cumulative node qualification and floor-pruned top-N links."""
    if not 0 <= edge_floor_fraction <= 1:
        raise ValueError("edge_floor_fraction must lie between zero and one")
    if profile is None:
        profile = _cumulative_top_n_profile(network, n_neighbors)
    edges = network["edges"]
    top_count, top_sum = profile["top_count"], profile["top_sum"]
    qualifies = (top_count == n_neighbors) & (top_sum >= threshold)
    selected_edge_ids = np.unique(profile["top_edge_id"][
        qualifies[profile["top_node"]]
    ])
    selected = edges.iloc[selected_edge_ids]
    induced_mask = (
        qualifies[selected.u.to_numpy(np.int64)]
        & qualifies[selected.v.to_numpy(np.int64)]
        & selected.F_ij.ge(edge_floor_fraction * float(threshold)).to_numpy()
    )
    return top_count, top_sum, qualifies, selected_edge_ids[induced_mask]


def cumulative_top_n_component_state(network: dict, threshold: float,
                                     analysis: pd.DataFrame,
                                     n_neighbors: int = 2,
                                     edge_floor_fraction: float =
                                     CUMULATIVE_EDGE_FLOOR_FRACTION) -> dict:
    """Form components from qualifying buildings' cumulative top-N links.

    Each building is scored by the sum of its N largest incident pairwise
    couplings. It qualifies only when it has N distinct neighbours and that
    cumulative score reaches ``threshold``. Components use the union of the
    qualifying buildings' selected top-N links, restricted to links whose two
    endpoints qualify and whose individual coupling is at least
    ``edge_floor_fraction * threshold``. All other within-perimeter buildings
    remain singleton nodes, matching the single-emitter analysis population.
    """
    nodes = network["nodes"].copy()
    edges = network["edges"]
    top_count, top_sum, qualifies, induced_edge_ids = _cumulative_top_n_selection(
        network, threshold, n_neighbors,
        edge_floor_fraction=edge_floor_fraction,
    )
    active_edges = edges.iloc[induced_edge_ids].copy()
    labels = _labels_after_active_edges(
        len(nodes), active_edges.u.to_numpy(), active_edges.v.to_numpy()
    )
    roots, sizes = np.unique(labels, return_counts=True)
    size_lookup = pd.Series(sizes, index=roots)
    nodes["top_n_neighbor_count"] = top_count
    nodes["top_n_sum"] = top_sum
    nodes["qualifies_cumulative_n"] = qualifies
    nodes["component_id"] = labels
    nodes["component_size"] = size_lookup.loc[labels].to_numpy(int)
    outcomes = (analysis[analysis.fire.eq(network["fire"])]
                .assign(BLD_ID=lambda frame: frame.BLD_ID.astype(str))
                .drop_duplicates("BLD_ID")
                .set_index("BLD_ID")["is_destroyed"])
    nodes["is_destroyed"] = nodes.BLD_ID.map(outcomes)
    nodes["assessed"] = nodes.is_destroyed.notna()

    if len(active_edges):
        active_roots = labels[active_edges.u.to_numpy()]
        component_mean_ssd = (
            pd.Series(active_edges.ssd_m.to_numpy())
            .groupby(active_roots).mean()
        )
        mean_component_ssd_ft = float(component_mean_ssd.mean() * M_TO_FT)
    else:
        component_mean_ssd = pd.Series(dtype=float)
        mean_component_ssd_ft = np.nan
    summary = {
        "fire": network["fire"], "cumulative_F_threshold": threshold,
        "n_neighbors": n_neighbors,
        "edge_floor_fraction": edge_floor_fraction,
        "individual_edge_floor": edge_floor_fraction * threshold,
        "nodes_inside_perimeter": len(nodes),
        "assessed_inside_perimeter": int(nodes.assessed.sum()),
        "destroyed_inside_perimeter": int(nodes.is_destroyed.fillna(0).sum()),
        "qualifying_buildings": int(qualifies.sum()),
        "active_bonds": len(active_edges), "components": len(roots),
        "connected_components": int(len(component_mean_ssd)),
        "connected_buildings": int(nodes.component_size.gt(1).sum()),
        "largest_component": int(sizes.max()),
        "mean_component_ssd_ft": mean_component_ssd_ft,
    }
    return {"fire": network["fire"], "threshold": threshold,
            "n_neighbors": n_neighbors,
            "edge_floor_fraction": edge_floor_fraction, "nodes": nodes,
            "active_edges": active_edges, "labels": labels,
            "summary": summary}


def top_n_eeat_component_state(network: dict, threshold: float,
                               analysis: pd.DataFrame,
                               n_neighbors: int = 2) -> dict:
    """Apply an individual EEAT threshold after Top-N edge pruning.

    Each node contributes at most its N strongest incident pairwise edges.
    The union of those selected edges is then filtered individually by
    ``F_ij >= threshold``. Unlike cumulative Top-N, edge weights are never
    summed and neither endpoint must first satisfy a node-level score.
    """
    nodes = network["nodes"].copy()
    edges = network["edges"]
    profile = _cumulative_top_n_profile(network, n_neighbors)
    selected_edge_ids = np.unique(profile["top_edge_id"])
    selected = edges.iloc[selected_edge_ids]
    active_edges = selected.loc[selected.F_ij.ge(threshold)].copy()
    labels = _labels_after_active_edges(
        len(nodes), active_edges.u.to_numpy(), active_edges.v.to_numpy()
    )
    roots, sizes = np.unique(labels, return_counts=True)
    size_lookup = pd.Series(sizes, index=roots)
    nodes["top_n_neighbor_count"] = profile["top_count"]
    nodes["component_id"] = labels
    nodes["component_size"] = size_lookup.loc[labels].to_numpy(int)
    outcomes = (analysis[analysis.fire.eq(network["fire"])]
                .assign(BLD_ID=lambda frame: frame.BLD_ID.astype(str))
                .drop_duplicates("BLD_ID")
                .set_index("BLD_ID")["is_destroyed"])
    nodes["is_destroyed"] = nodes.BLD_ID.map(outcomes)
    nodes["assessed"] = nodes.is_destroyed.notna()

    if len(active_edges):
        active_roots = labels[active_edges.u.to_numpy()]
        component_mean_ssd = (
            pd.Series(active_edges.ssd_m.to_numpy())
            .groupby(active_roots).mean()
        )
        mean_component_ssd_ft = float(component_mean_ssd.mean() * M_TO_FT)
    else:
        component_mean_ssd = pd.Series(dtype=float)
        mean_component_ssd_ft = np.nan
    summary = {
        "fire": network["fire"], "F_ij_threshold": threshold,
        "n_neighbors": n_neighbors, "clustering_method": "top_n_pruned_eeat",
        "nodes_inside_perimeter": len(nodes),
        "assessed_inside_perimeter": int(nodes.assessed.sum()),
        "destroyed_inside_perimeter": int(nodes.is_destroyed.fillna(0).sum()),
        "candidate_top_n_bonds": len(selected),
        "active_bonds": len(active_edges), "components": len(roots),
        "connected_components": int(len(component_mean_ssd)),
        "connected_buildings": int(nodes.component_size.gt(1).sum()),
        "largest_component": int(sizes.max()),
        "mean_component_ssd_ft": mean_component_ssd_ft,
    }
    return {"fire": network["fire"], "threshold": threshold,
            "n_neighbors": n_neighbors, "nodes": nodes,
            "active_edges": active_edges, "labels": labels,
            "summary": summary}


def top_n_pruned_network(network: dict, n_neighbors: int = 2) -> dict:
    """Restrict a network to the union of each node's N strongest edges.

    This is the selection ``top_n_eeat_component_state`` applies before its
    individual threshold. It does not depend on the threshold, so passing the
    pruned network to ``shared_fate_curve`` rebuilds Top-N-pruned EEAT
    components at every cutoff of a filtration.
    """
    profile = _cumulative_top_n_profile(network, n_neighbors)
    selected_edge_ids = np.unique(profile["top_edge_id"])
    return {**network,
            "edges": network["edges"].iloc[selected_edge_ids]
            .reset_index(drop=True)}


ANCHORED_EDGE_FLOOR_FRACTION = 0.75


def anchored_eeat_edge_mask(network: dict, threshold: float,
                            alpha: float = ANCHORED_EDGE_FLOOR_FRACTION,
                            require_both: bool = False) -> np.ndarray:
    """Return the edges kept by anchored EEAT.

    Every edge with ``F_ij >= threshold`` is kept, as in EEAT. An edge with
    ``alpha * threshold <= F_ij < threshold`` is also kept when an endpoint
    has at least one other edge at or above ``threshold``: either endpoint by
    default, both with ``require_both``. A weaker link is admitted only where
    it is anchored by a building that is already strongly coupled, and there
    is no limit on the number of links per building.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie between zero and one")
    edges = network["edges"]
    coupling = edges.F_ij.to_numpy(float)
    u, v = edges.u.to_numpy(np.int64), edges.v.to_numpy(np.int64)
    n_nodes = len(network["nodes"])
    strong = coupling >= threshold
    strong_degree = (np.bincount(u[strong], minlength=n_nodes)
                     + np.bincount(v[strong], minlength=n_nodes))
    anchored_u, anchored_v = strong_degree[u] > 0, strong_degree[v] > 0
    anchor = (anchored_u & anchored_v) if require_both else (anchored_u | anchored_v)
    weak = (coupling >= alpha * threshold) & ~strong
    return strong | (weak & anchor)


def anchored_eeat_component_state(network: dict, threshold: float,
                                  analysis: pd.DataFrame,
                                  alpha: float = ANCHORED_EDGE_FLOOR_FRACTION,
                                  require_both: bool = False) -> dict:
    """Form anchored-EEAT components and attach outcomes."""
    nodes = network["nodes"].copy()
    edges = network["edges"]
    keep = anchored_eeat_edge_mask(network, threshold, alpha, require_both)
    active_edges = edges.loc[keep].copy()
    labels = _labels_after_active_edges(
        len(nodes), active_edges.u.to_numpy(), active_edges.v.to_numpy()
    )
    roots, sizes = np.unique(labels, return_counts=True)
    size_lookup = pd.Series(sizes, index=roots)
    nodes["component_id"] = labels
    nodes["component_size"] = size_lookup.loc[labels].to_numpy(int)
    outcomes = (analysis[analysis.fire.eq(network["fire"])]
                .assign(BLD_ID=lambda frame: frame.BLD_ID.astype(str))
                .drop_duplicates("BLD_ID")
                .set_index("BLD_ID")["is_destroyed"])
    nodes["is_destroyed"] = nodes.BLD_ID.map(outcomes)
    nodes["assessed"] = nodes.is_destroyed.notna()
    strong_bonds = int(active_edges.F_ij.ge(threshold).sum())
    summary = {
        "fire": network["fire"], "F_ij_threshold": threshold,
        "alpha": alpha, "anchor_rule": "both" if require_both else "either",
        "clustering_method": "anchored_eeat",
        "nodes_inside_perimeter": len(nodes),
        "assessed_inside_perimeter": int(nodes.assessed.sum()),
        "destroyed_inside_perimeter": int(nodes.is_destroyed.fillna(0).sum()),
        "strong_bonds": strong_bonds,
        "anchored_bonds": len(active_edges) - strong_bonds,
        "active_bonds": len(active_edges), "components": len(roots),
        "connected_components": int((sizes > 1).sum()),
        "connected_buildings": int(nodes.component_size.gt(1).sum()),
        "largest_component": int(sizes.max()),
    }
    return {"fire": network["fire"], "threshold": threshold, "alpha": alpha,
            "nodes": nodes, "active_edges": active_edges, "labels": labels,
            "summary": summary}


def selected_edge_shared_fate_curve(
        networks: dict[str, dict], analysis: pd.DataFrame, fit: dict,
        select_edges, probabilities=None, n_shuffle: int = 499,
        seed: int = 20250804) -> pd.DataFrame:
    """Within-SEN outcome diversity for any edge-selection rule.

    ``select_edges(network, cutoff)`` returns a boolean mask of active edges.
    Shuffles follow ``shared_fate_curve`` draw for draw, so passing the EEAT
    rule ``F_ij >= cutoff`` reproduces it exactly.
    """
    if probabilities is None:
        probabilities = np.arange(.05, .901, .025)
    probabilities = np.asarray(probabilities, float)
    cutoffs = np.asarray([coupling_at_probability(fit, p)
                          for p in probabilities])
    store = {}
    for fire in FIRES:
        network = networks[fire]
        nodes = network["nodes"]
        outcomes = (analysis[analysis.fire.eq(fire)]
                    .assign(BLD_ID=lambda frame: frame.BLD_ID.astype(str))
                    .drop_duplicates("BLD_ID").set_index("BLD_ID")["is_destroyed"])
        fate = nodes.BLD_ID.map(outcomes)
        assessed = fate.notna().to_numpy()
        u = network["edges"].u.to_numpy()
        v = network["edges"].v.to_numpy()
        labels_by_cutoff = []
        for cutoff in cutoffs:
            keep = select_edges(network, float(cutoff))
            labels = _labels_after_active_edges(len(nodes), u[keep], v[keep])
            labels_by_cutoff.append(labels[assessed])
        store[fire] = {"N": len(nodes),
                       "destroyed": fate.fillna(0).to_numpy(float)[assessed],
                       "labels": labels_by_cutoff}

    observed_num = {fire: [] for fire in FIRES}
    observed_weight = {fire: [] for fire in FIRES}
    for fire in FIRES:
        for labels in store[fire]["labels"]:
            numerator, weight = _diversity_numerator(
                labels, store[fire]["destroyed"], store[fire]["N"])
            observed_num[fire].append(numerator)
            observed_weight[fire].append(weight)
    rng = np.random.default_rng(seed)
    null = np.empty((n_shuffle, len(cutoffs)))
    for draw in range(n_shuffle):
        total_num = np.zeros(len(cutoffs))
        total_weight = np.zeros(len(cutoffs))
        for fire in FIRES:
            shuffled = rng.permutation(store[fire]["destroyed"])
            for index, labels in enumerate(store[fire]["labels"]):
                numerator, weight = _diversity_numerator(
                    labels, shuffled, store[fire]["N"])
                total_num[index] += numerator
                total_weight[index] += weight
        null[draw] = total_num / total_weight

    rows = []
    pooled_num = np.sum([observed_num[fire] for fire in FIRES], axis=0)
    pooled_weight = np.sum([observed_weight[fire] for fire in FIRES], axis=0)
    pooled_obs = pooled_num / pooled_weight
    null_lo, null_hi = np.quantile(null, [.025, .975], axis=0)
    for index, (probability, cutoff) in enumerate(zip(probabilities, cutoffs)):
        for fire in FIRES:
            rows.append({
                "fire": fire, "p_destroyed_equivalent": probability,
                "F_ij_cutoff": cutoff,
                "observed_diversity": (observed_num[fire][index] /
                                       observed_weight[fire][index]),
                "null_mean": np.nan, "null_lo": np.nan, "null_hi": np.nan,
            })
        rows.append({
            "fire": "POOLED", "p_destroyed_equivalent": probability,
            "F_ij_cutoff": cutoff, "observed_diversity": pooled_obs[index],
            "null_mean": null[:, index].mean(),
            "null_lo": null_lo[index], "null_hi": null_hi[index],
        })
    return pd.DataFrame(rows)


def minimum_neighbors_to_threshold(network: dict,
                                   threshold: float) -> pd.DataFrame:
    """Return the minimum number of strongest incident pairs needed per node.

    Incident pairwise ``F_ij`` values are sorted from largest to smallest for
    each building and cumulatively summed. ``min_neighbors_to_threshold`` is
    the first rank at which that sum reaches ``threshold``. It is nullable for
    buildings whose total incident coupling remains below the threshold.
    """
    nodes = network["nodes"][["BLD_ID", "node", "geometry"]].copy()
    edges = network["edges"]
    incidence = pd.concat([
        pd.DataFrame({"node": edges.u.to_numpy(np.int64),
                      "F_ij": edges.F_ij.to_numpy(float)}),
        pd.DataFrame({"node": edges.v.to_numpy(np.int64),
                      "F_ij": edges.F_ij.to_numpy(float)}),
    ], ignore_index=True)
    incidence = incidence.sort_values(
        ["node", "F_ij"], ascending=[True, False], kind="stable"
    )
    incidence["neighbor_rank"] = (
        incidence.groupby("node", sort=False).cumcount() + 1
    )
    incidence["cumulative_F_score"] = (
        incidence.groupby("node", sort=False).F_ij.cumsum()
    )
    first_reaching = (
        incidence[incidence.cumulative_F_score.ge(threshold)]
        .drop_duplicates("node")
        .set_index("node")
    )
    degree = incidence.groupby("node", sort=False).size()
    total_incident = incidence.groupby("node", sort=False).F_ij.sum()
    nodes["available_neighbors"] = (
        nodes.node.map(degree).fillna(0).astype(np.int32)
    )
    nodes["total_incident_F_score"] = (
        nodes.node.map(total_incident).fillna(0.0).astype(float)
    )
    nodes["min_neighbors_to_threshold"] = pd.array(
        nodes.node.map(first_reaching.neighbor_rank), dtype="Int32"
    )
    nodes["F_score_at_min_neighbors"] = nodes.node.map(
        first_reaching.cumulative_F_score
    )
    nodes["threshold_reachable"] = nodes.min_neighbors_to_threshold.notna()
    nodes["F_score_threshold"] = float(threshold)
    return nodes


def _wilson(successes: int, n: int) -> tuple[float, float]:
    """Wilson 95% interval, clamped to contain the estimate.

    The interval contains the sample share mathematically, but at a share of
    0 or 1 floating point can leave the bound a hair inside it, which makes
    error bars negative.
    """
    lo, hi = proportion_confint(successes, n, method="wilson")
    share = successes / n
    return min(lo, share), max(hi, share)


def component_size_classes(states: dict[str, dict], n_quantiles: int = 6
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign fire-specific building-weighted size classes and summarize loss."""
    node_parts, rows = [], []
    for fire in FIRES:
        nodes = states[fire]["nodes"].copy()
        assessed_connected = nodes.loc[
            nodes.assessed & nodes.component_size.gt(1), "component_size"
        ].to_numpy()
        upper = np.unique(np.ceil(np.quantile(
            assessed_connected, np.linspace(1 / n_quantiles, 1, n_quantiles)
        )).astype(int))
        if len(upper) != n_quantiles:
            raise RuntimeError(f"{fire}: size quantiles collapsed to {len(upper)} classes")
        nodes["size_class_id"] = np.where(
            nodes.component_size.eq(1), 0,
            1 + np.searchsorted(upper, nodes.component_size, side="left")
        )
        nodes["size_class"] = nodes.size_class_id.map(
            {0: "Isolated", **{index: f"Q{index}" for index in range(1, 7)}}
        )
        node_parts.append(nodes.assign(fire=fire))
        assessed = nodes[nodes.assessed].copy()
        for class_id in range(7):
            group = assessed[assessed.size_class_id.eq(class_id)]
            if group.empty:
                continue
            low, high = group.component_size.min(), group.component_size.max()
            successes = int(group.is_destroyed.sum())
            lo, hi = _wilson(successes, len(group))
            rows.append({
                "fire": fire, "size_class_id": class_id,
                "size_class": "Isolated" if class_id == 0 else f"Q{class_id}",
                "component_size_range": (str(low) if low == high
                                         else f"{low}-{high}"),
                "assessed": len(group), "destroyed": successes,
                "destroyed_share": successes / len(group),
                "ci_lo": lo, "ci_hi": hi,
            })
    node_table = pd.concat(node_parts, ignore_index=True)
    by_fire = pd.DataFrame(rows)
    pooled = (by_fire.groupby(["size_class_id", "size_class"], as_index=False)
              .agg(assessed=("assessed", "sum"),
                   destroyed=("destroyed", "sum")))
    pooled["fire"] = "POOLED"
    pooled["component_size_range"] = "Fire-specific quantiles"
    pooled["destroyed_share"] = pooled.destroyed / pooled.assessed
    intervals = [_wilson(int(row.destroyed), int(row.assessed))
                 for row in pooled.itertuples(index=False)]
    pooled[["ci_lo", "ci_hi"]] = intervals
    summary = pd.concat([by_fire, pooled[by_fire.columns]], ignore_index=True)
    return node_table, summary.sort_values(["fire", "size_class_id"])


def class_shuffle_analysis(class_summary: pd.DataFrame, n_shuffle: int = 9999,
                           seed: int = 20250803
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle destruction within fire while preserving class sizes and totals."""
    rng = np.random.default_rng(seed)
    draws, null_rows, contrast_rows = {}, [], []
    for fire in FIRES:
        frame = class_summary[class_summary.fire.eq(fire)].sort_values("size_class_id")
        class_n = frame.assessed.to_numpy(int)
        total_destroyed = int(frame.destroyed.sum())
        destroyed_draws = rng.multivariate_hypergeometric(
            class_n, total_destroyed, size=n_shuffle
        )
        share_draws = destroyed_draws / class_n[None, :]
        draws[fire] = destroyed_draws
        for position, row in enumerate(frame.itertuples(index=False)):
            null_rows.append({
                "fire": fire, "size_class_id": row.size_class_id,
                "size_class": row.size_class,
                "null_mean": share_draws[:, position].mean(),
                "null_lo": np.quantile(share_draws[:, position], .025),
                "null_hi": np.quantile(share_draws[:, position], .975),
            })
        observed = frame.destroyed_share.to_numpy()
        null_delta = share_draws[:, -1] - share_draws[:, 0]
        observed_delta = observed[-1] - observed[0]
        p_value = ((null_delta >= observed_delta).sum() + 1) / (n_shuffle + 1)
        contrast_rows.append({
            "fire": fire, "contrast": "Q6 minus isolated",
            "difference_pp": 100 * observed_delta,
            "shuffle_p": p_value, "n_shuffle": n_shuffle,
        })

    pooled = class_summary[class_summary.fire.eq("POOLED")].sort_values("size_class_id")
    pooled_n = pooled.assessed.to_numpy(int)
    pooled_destroyed = draws["EATON"] + draws["PALISADES"]
    pooled_share = pooled_destroyed / pooled_n[None, :]
    for position, row in enumerate(pooled.itertuples(index=False)):
        null_rows.append({
            "fire": "POOLED", "size_class_id": row.size_class_id,
            "size_class": row.size_class,
            "null_mean": pooled_share[:, position].mean(),
            "null_lo": np.quantile(pooled_share[:, position], .025),
            "null_hi": np.quantile(pooled_share[:, position], .975),
        })
    observed = pooled.destroyed_share.to_numpy()
    null_delta = pooled_share[:, -1] - pooled_share[:, 0]
    observed_delta = observed[-1] - observed[0]
    contrast_rows.append({
        "fire": "POOLED", "contrast": "Q6 minus isolated",
        "difference_pp": 100 * observed_delta,
        "shuffle_p": ((null_delta >= observed_delta).sum() + 1) /
                     (n_shuffle + 1),
        "n_shuffle": n_shuffle,
    })
    return pd.DataFrame(null_rows), pd.DataFrame(contrast_rows)


def _diversity_numerator(labels: np.ndarray, destroyed: np.ndarray,
                         n_nodes: int) -> tuple[float, float]:
    counts = np.bincount(labels, minlength=n_nodes).astype(float)
    multi = counts >= 2
    destroyed_count = np.bincount(
        labels, weights=destroyed, minlength=n_nodes
    )
    probability = np.zeros(n_nodes)
    probability[multi] = destroyed_count[multi] / counts[multi]
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(probability * np.log2(probability) +
                    (1 - probability) * np.log2(1 - probability))
    entropy[(probability <= 0) | (probability >= 1)] = 0
    return float((counts[multi] * entropy[multi]).sum()), float(counts[multi].sum())


def shared_fate_curve(networks: dict[str, dict], analysis: pd.DataFrame,
                      fit: dict, probabilities=None,
                      n_shuffle: int = 499, seed: int = 20250804
                      ) -> pd.DataFrame:
    """Compare within-SEN Shannon outcome diversity with label shuffles."""
    if probabilities is None:
        probabilities = np.arange(.05, .901, .025)
    probabilities = np.asarray(probabilities, float)
    cutoffs = np.asarray([coupling_at_probability(fit, p)
                          for p in probabilities])
    store = {}
    for fire in FIRES:
        network = networks[fire]
        nodes = network["nodes"]
        outcomes = (analysis[analysis.fire.eq(fire)]
                    .assign(BLD_ID=lambda frame: frame.BLD_ID.astype(str))
                    .drop_duplicates("BLD_ID").set_index("BLD_ID")["is_destroyed"])
        fate = nodes.BLD_ID.map(outcomes)
        assessed = fate.notna().to_numpy()
        destroyed = fate.fillna(0).to_numpy(float)[assessed]
        edge_order = np.argsort(-network["edges"].F_ij.to_numpy(), kind="stable")
        sorted_weight = network["edges"].F_ij.to_numpy()[edge_order]
        labels_by_cutoff = []
        for cutoff in cutoffs:
            n_active = int(np.count_nonzero(sorted_weight >= cutoff))
            selected = edge_order[:n_active]
            labels = _labels_after_active_edges(
                len(nodes), network["edges"].u.to_numpy()[selected],
                network["edges"].v.to_numpy()[selected]
            )
            labels_by_cutoff.append(labels[assessed])
        store[fire] = {
            "N": len(nodes), "destroyed": destroyed,
            "labels": labels_by_cutoff,
        }

    observed_num = {fire: [] for fire in FIRES}
    observed_weight = {fire: [] for fire in FIRES}
    for fire in FIRES:
        for labels in store[fire]["labels"]:
            numerator, weight = _diversity_numerator(
                labels, store[fire]["destroyed"], store[fire]["N"]
            )
            observed_num[fire].append(numerator)
            observed_weight[fire].append(weight)
    rng = np.random.default_rng(seed)
    null = np.empty((n_shuffle, len(cutoffs)))
    for draw in range(n_shuffle):
        total_num = np.zeros(len(cutoffs))
        total_weight = np.zeros(len(cutoffs))
        for fire in FIRES:
            shuffled = rng.permutation(store[fire]["destroyed"])
            for index, labels in enumerate(store[fire]["labels"]):
                numerator, weight = _diversity_numerator(
                    labels, shuffled, store[fire]["N"]
                )
                total_num[index] += numerator
                total_weight[index] += weight
        null[draw] = total_num / total_weight

    rows = []
    pooled_num = np.sum([observed_num[fire] for fire in FIRES], axis=0)
    pooled_weight = np.sum([observed_weight[fire] for fire in FIRES], axis=0)
    pooled_obs = pooled_num / pooled_weight
    null_lo, null_hi = np.quantile(null, [.025, .975], axis=0)
    for index, (probability, cutoff) in enumerate(zip(probabilities, cutoffs)):
        for fire in FIRES:
            rows.append({
                "fire": fire, "p_destroyed_equivalent": probability,
                "F_ij_cutoff": cutoff,
                "observed_diversity": (observed_num[fire][index] /
                                       observed_weight[fire][index]),
                "null_mean": np.nan, "null_lo": np.nan, "null_hi": np.nan,
            })
        rows.append({
            "fire": "POOLED", "p_destroyed_equivalent": probability,
            "F_ij_cutoff": cutoff, "observed_diversity": pooled_obs[index],
            "null_mean": null[:, index].mean(),
            "null_lo": null_lo[index], "null_hi": null_hi[index],
        })
    return pd.DataFrame(rows)


def cumulative_top_n_shared_fate_curve(
        networks: dict[str, dict], analysis: pd.DataFrame, fit: dict,
        n_neighbors: int = 2, probabilities=None,
        n_shuffle: int = 499, seed: int = 20250804,
        edge_floor_fraction: float =
        CUMULATIVE_EDGE_FLOOR_FRACTION) -> pd.DataFrame:
    """Rebuild cumulative top-N components at every fragility-equivalent cutoff."""
    if probabilities is None:
        probabilities = np.arange(.05, .901, .025)
    probabilities = np.asarray(probabilities, float)
    cutoffs = np.asarray([
        coupling_at_probability(fit, probability)
        for probability in probabilities
    ])
    store = {}
    for fire in FIRES:
        network = networks[fire]
        nodes = network["nodes"]
        outcomes = (analysis[analysis.fire.eq(fire)]
                    .assign(BLD_ID=lambda frame: frame.BLD_ID.astype(str))
                    .drop_duplicates("BLD_ID")
                    .set_index("BLD_ID")["is_destroyed"])
        fate = nodes.BLD_ID.map(outcomes)
        assessed = fate.notna().to_numpy()
        destroyed = fate.fillna(0).to_numpy(float)[assessed]
        top_n_profile = _cumulative_top_n_profile(network, n_neighbors)
        labels_by_cutoff = []
        network_by_cutoff = []
        for cutoff in cutoffs:
            _, _, qualifies, induced_edge_ids = _cumulative_top_n_selection(
                network, float(cutoff), n_neighbors, profile=top_n_profile,
                edge_floor_fraction=edge_floor_fraction,
            )
            active_edges = network["edges"].iloc[induced_edge_ids]
            labels = _labels_after_active_edges(
                len(nodes), active_edges.u.to_numpy(), active_edges.v.to_numpy()
            )
            labels_by_cutoff.append(labels[assessed])
            component_sizes = np.bincount(labels)[labels]
            network_by_cutoff.append({
                "qualifying_buildings": int(qualifies.sum()),
                "active_bonds": len(induced_edge_ids),
                "connected_buildings": int(np.count_nonzero(component_sizes > 1)),
            })
        store[fire] = {
            "N": len(nodes), "destroyed": destroyed,
            "labels": labels_by_cutoff, "network": network_by_cutoff,
        }

    observed_num = {fire: [] for fire in FIRES}
    observed_weight = {fire: [] for fire in FIRES}
    for fire in FIRES:
        for labels in store[fire]["labels"]:
            numerator, weight = _diversity_numerator(
                labels, store[fire]["destroyed"], store[fire]["N"]
            )
            observed_num[fire].append(numerator)
            observed_weight[fire].append(weight)
    rng = np.random.default_rng(seed)
    null = np.full((n_shuffle, len(cutoffs)), np.nan)
    for draw in range(n_shuffle):
        total_num = np.zeros(len(cutoffs))
        total_weight = np.zeros(len(cutoffs))
        for fire in FIRES:
            shuffled = rng.permutation(store[fire]["destroyed"])
            for index, labels in enumerate(store[fire]["labels"]):
                numerator, weight = _diversity_numerator(
                    labels, shuffled, store[fire]["N"]
                )
                total_num[index] += numerator
                total_weight[index] += weight
        np.divide(total_num, total_weight, out=null[draw],
                  where=total_weight > 0)

    rows = []
    pooled_num = np.sum([observed_num[fire] for fire in FIRES], axis=0)
    pooled_weight = np.sum([observed_weight[fire] for fire in FIRES], axis=0)
    pooled_obs = np.divide(
        pooled_num, pooled_weight,
        out=np.full(len(cutoffs), np.nan), where=pooled_weight > 0,
    )
    null_lo, null_hi = np.nanquantile(null, [.025, .975], axis=0)
    for index, (probability, cutoff) in enumerate(zip(probabilities, cutoffs)):
        for fire in FIRES:
            weight = observed_weight[fire][index]
            network_metrics = store[fire]["network"][index]
            rows.append({
                "fire": fire, "n_neighbors": n_neighbors,
                "p_destroyed_equivalent": probability,
                "F_score_cutoff": cutoff, "F_ij_cutoff": cutoff,
                **network_metrics,
                "observed_diversity": (
                    observed_num[fire][index] / weight if weight else np.nan
                ),
                "null_mean": np.nan, "null_lo": np.nan, "null_hi": np.nan,
            })
        rows.append({
            "fire": "POOLED", "n_neighbors": n_neighbors,
            "p_destroyed_equivalent": probability,
            "F_score_cutoff": cutoff, "F_ij_cutoff": cutoff,
            "qualifying_buildings": sum(
                store[fire]["network"][index]["qualifying_buildings"]
                for fire in FIRES
            ),
            "active_bonds": sum(
                store[fire]["network"][index]["active_bonds"]
                for fire in FIRES
            ),
            "connected_buildings": sum(
                store[fire]["network"][index]["connected_buildings"]
                for fire in FIRES
            ),
            "observed_diversity": pooled_obs[index],
            "null_mean": np.nanmean(null[:, index]),
            "null_lo": null_lo[index], "null_hi": null_hi[index],
        })
    return pd.DataFrame(rows)


def continuous_component_outcomes(states: dict[str, dict]) -> pd.DataFrame:
    """Return assessed node outcomes with their exact SEN component size."""
    rows = []
    for fire in FIRES:
        nodes = states[fire]["nodes"]
        assessed = nodes[nodes.assessed].copy()
        rows.append(pd.DataFrame({
            "fire": fire, "component_id": assessed.component_id,
            "component_size": assessed.component_size,
            "is_destroyed": assessed.is_destroyed.astype(int),
        }))
    return pd.concat(rows, ignore_index=True)


def component_size_count_summary(states: dict[str, dict],
                                 n_shuffle: int = 9999,
                                 seed: int = 20250803) -> pd.DataFrame:
    """Summarize destruction over absolute SEN-size bins and a shuffle null.

    Component sizes are grouped into doubling bins (1, 2, 3--4, 5--8, ...),
    with points positioned at the geometric mean observed component size.
    The null redistributes each fire's destroyed total across the same bins
    while preserving the assessed count in every bin.
    """
    outcomes = continuous_component_outcomes(states)
    size_bins = [1]
    while size_bins[-1] < outcomes.component_size.max():
        size_bins.append(2 * size_bins[-1])
    size_bins = np.asarray(size_bins, dtype=int)
    size_labels = [
        str(upper) if upper <= 2 else f"{upper // 2 + 1}-{upper}"
        for upper in size_bins
    ]
    binned = outcomes.assign(
        size_bin=np.searchsorted(
            size_bins, outcomes.component_size.to_numpy(), side="left"
        )
    )

    rows = []
    for label, frame in (
        [(fire, binned[binned.fire.eq(fire)]) for fire in FIRES]
        + [("POOLED", binned)]
    ):
        for bin_id, group in frame.groupby("size_bin"):
            destroyed, assessed = int(group.is_destroyed.sum()), len(group)
            lo, hi = _wilson(destroyed, assessed)
            rows.append({
                "fire": label,
                "size_bin_id": int(bin_id),
                "component_size_range": size_labels[bin_id],
                "bin_upper": int(size_bins[bin_id]),
                "size_position": float(np.exp(
                    np.log(group.component_size.to_numpy()).mean()
                )),
                "assessed": assessed,
                "destroyed": destroyed,
                "destroyed_share": destroyed / assessed,
                "ci_lo": lo,
                "ci_hi": hi,
            })
    summary = pd.DataFrame(rows)

    rng = np.random.default_rng(seed)
    fire_draws, null_rows = {}, []
    for fire in FIRES:
        frame = summary[summary.fire.eq(fire)].sort_values("size_bin_id")
        draws = rng.multivariate_hypergeometric(
            frame.assessed.to_numpy(int), int(frame.destroyed.sum()),
            size=n_shuffle,
        )
        fire_draws[fire] = pd.DataFrame(
            draws, columns=frame.size_bin_id.to_numpy()
        )
        shares = draws / frame.assessed.to_numpy(int)[None, :]
        for position, row in enumerate(frame.itertuples(index=False)):
            null_rows.append({
                "fire": fire,
                "size_bin_id": row.size_bin_id,
                "null_mean": shares[:, position].mean(),
                "null_lo": np.quantile(shares[:, position], .025),
                "null_hi": np.quantile(shares[:, position], .975),
            })

    pooled = summary[summary.fire.eq("POOLED")].sort_values("size_bin_id")
    pooled_counts = (
        fire_draws["EATON"].add(fire_draws["PALISADES"], fill_value=0)
        .reindex(columns=pooled.size_bin_id.to_numpy(), fill_value=0)
        .to_numpy()
    )
    pooled_shares = pooled_counts / pooled.assessed.to_numpy(int)[None, :]
    for position, row in enumerate(pooled.itertuples(index=False)):
        null_rows.append({
            "fire": "POOLED",
            "size_bin_id": row.size_bin_id,
            "null_mean": pooled_shares[:, position].mean(),
            "null_lo": np.quantile(pooled_shares[:, position], .025),
            "null_hi": np.quantile(pooled_shares[:, position], .975),
        })
    return summary.merge(
        pd.DataFrame(null_rows), on=["fire", "size_bin_id"], how="left"
    ).sort_values(["fire", "size_bin_id"])


def distance_pairs(geometries, distance_m: float):
    """Positional index pairs ``(i, j)``, ``i < j``, of footprints within
    ``distance_m`` of each other, edge to edge."""
    import shapely

    geometries = np.asarray(geometries)
    i, j = shapely.STRtree(geometries).query(
        geometries, predicate="dwithin", distance=distance_m)
    keep = i < j
    return i[keep], j[keep]


def distance_components(geometries, distance_m: float):
    """Connected components when every footprint pair within ``distance_m``
    (edge to edge) is linked: a geometry-only reference SEN.

    Returns component labels, component sizes per geometry and the number of
    linked pairs.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    geometries = np.asarray(geometries)
    i, j = distance_pairs(geometries, distance_m)
    graph = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(len(geometries),) * 2)
    _, labels = connected_components(graph, directed=False)
    return labels, np.bincount(labels)[labels], len(i)
