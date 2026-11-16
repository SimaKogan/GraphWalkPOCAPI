# save as: print_tree.py
from pathlib import Path

EXCLUDE_DIRS = {
    '.git', '.idea', '.vscode', '__pycache__', 'node_modules',
    'venv', '.venv', 'env', '.env', '.mypy_cache', '.pytest_cache',
    'dist', 'build', '.cache'
}

# Path-specific exclusions relative to project root (case-insensitive, POSIX-style)
EXCLUDE_PATHS = {
    'Resources/Traceability'.lower()
}


def _is_excluded_dir(dir_path: Path, base: Path) -> bool:
    if dir_path.name in EXCLUDE_DIRS:
        return True
    rel = dir_path.relative_to(base).as_posix().lower()
    return any(rel == ex or rel.startswith(ex + '/') for ex in EXCLUDE_PATHS)


def print_tree(root: Path, prefix: str = "", base: Path | None = None):
    if base is None:
        base = root
    items = []
    for p in root.iterdir():
        if (p.is_dir() and not _is_excluded_dir(p, base)) or p.is_file():
            items.append(p)
    items = sorted(items, key=lambda p: (not p.is_dir(), p.name.lower()))
    for i, p in enumerate(items):
        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + p.name)
        if p.is_dir() and not _is_excluded_dir(p, base):
            extension = "    " if i == len(items) - 1 else "│   "
            print_tree(p, prefix + extension, base=base)


if __name__ == "__main__":
    # Set your project root here:
    project_root = Path(__file__).resolve().parents[1]
    print(project_root.name)
    print_tree(project_root, base=project_root)
