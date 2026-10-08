"""Audit the code-only portion of the S2S reproduction package."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "s2s_nc" / "notebooks"
NOTEBOOKS = [
    NOTEBOOK_DIR / f"{index:02d}_{name}.ipynb"
    for index, name in enumerate((
        "population_sample",
        "coupling_and_fragility",
        "construction_materials",
        "defense",
        "sen",
        "rsen",
        "carsen",
    ))
]
FORBIDDEN_SOURCE = (
    "/Users/",
    "PACKAGE_ROOT.parent",
    "reviewer_reproduction.src",
    "src.features",
    "src.config",
)


def local_imports(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("src"):
                modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(
                alias.name for alias in node.names if alias.name.startswith("src")
            )
    return modules


def module_path(module: str) -> Path:
    return ROOT.joinpath(*module.split(".")).with_suffix(".py")


def main() -> None:
    problems: list[str] = []
    modules: set[str] = set()
    for path in NOTEBOOKS:
        if not path.exists():
            problems.append(f"missing notebook: {path.relative_to(ROOT)}")
            continue
        notebook = json.loads(path.read_text(encoding="utf-8"))
        kernel = notebook.get("metadata", {}).get("kernelspec", {}).get("name")
        if kernel != "s2s-fire-reproduction":
            problems.append(
                f"{path.name}: kernel is {kernel!r}, "
                "expected 's2s-fire-reproduction'"
            )
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            for marker in FORBIDDEN_SOURCE:
                if marker in source:
                    problems.append(
                        f"{path.name}:{cell.get('id', '?')}: forbidden {marker!r}"
                    )
            try:
                tree = ast.parse(source)
            except SyntaxError as error:
                problems.append(f"{path.name}:{cell.get('id', '?')}: {error}")
            else:
                modules.update(local_imports(tree))

    pending = list(modules)
    while pending:
        module = pending.pop()
        path = module_path(module)
        if not path.exists():
            problems.append(f"missing local module: {path.relative_to(ROOT)}")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as error:
            problems.append(f"{path.relative_to(ROOT)}: {error}")
            continue
        for dependency in local_imports(tree) - modules:
            modules.add(dependency)
            pending.append(dependency)

    if problems:
        print("S2S code validation failed:")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)
    print(
        f"S2S code validation passed: {len(NOTEBOOKS)} notebooks, "
        f"{len(modules)} local modules."
    )


if __name__ == "__main__":
    main()
