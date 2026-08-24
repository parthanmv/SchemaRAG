"""Regression guard: every third-party runtime import must be declared.

A fresh ``pip install -r requirements.txt`` environment must be able to
import and run the application. This test scans the actual imports under
``app/`` (via AST, so nothing is executed) and asserts each third-party
top-level package is listed in ``requirements.txt``. It exists because
``sqlglot`` was a hard runtime dependency of Phase 4 while missing from the
requirements file.
"""

from __future__ import annotations

import ast
import pathlib
import re

APP_DIR = pathlib.Path(__file__).resolve().parents[1] / "app"
REQUIREMENTS = APP_DIR.parent / "requirements.txt"

#: import name -> distribution name as pip knows it
ALIASES: dict[str, str] = {
    "google": "google-genai",
    "sentence_transformers": "sentence-transformers",
    "faiss": "faiss-cpu",
    "pydantic_settings": "pydantic-settings",
    "sklearn": "scikit-learn",
}

STDLIB_OR_LOCAL = {
    # stdlib (subset actually imported by app/); anything else unknown is
    # treated as third-party on purpose - failing loudly is the safe default.
    "__future__", "abc", "argparse", "ast", "asyncio", "collections",
    "contextlib", "dataclasses", "datetime", "decimal", "difflib", "enum",
    "functools", "getpass", "hashlib", "http", "importlib", "io", "itertools", "json",
    "logging", "math", "os", "pathlib", "random", "re", "secrets", "string", "sys",
    "threading", "time", "types", "typing", "urllib", "uuid", "warnings",
}


def _iter_py_files() -> list[pathlib.Path]:
    return sorted(APP_DIR.rglob("*.py"))


def _top_level_imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_requirements_declare_all_runtime_imports() -> None:
    declared_raw = REQUIREMENTS.read_text(encoding="utf-8").lower()
    # Strip comments/whitespace/extras to bare distribution names.
    declared = set()
    for line in declared_raw.splitlines():
        line = line.split("#")[0].strip()
        if line:
            declared.add(re.split(r"[<>=!\[;\s]", line, maxsplit=1)[0])

    imported: dict[str, list[str]] = {}
    for path in _iter_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _top_level_imports(tree):
            if name == "app" or name in STDLIB_OR_LOCAL:
                continue
            imported.setdefault(name, []).append(path.name)

    assert imported, "no imports discovered - scanner is broken"

    missing: list[str] = []
    for module_name in sorted(imported):
        distribution = ALIASES.get(module_name, module_name)
        if distribution.lower() not in declared:
            users = ", ".join(sorted(set(imported[module_name])))
            missing.append(f"{module_name} (as {distribution}) imported by {users}")

    assert not missing, (
        "runtime imports missing from requirements.txt: " + "; ".join(missing)
    )
