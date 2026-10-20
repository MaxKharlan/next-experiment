"""Copy artifact bytes to object-store mounts without POSIX metadata operations."""

import shutil
from pathlib import Path


def export_tree(source: Path, destination: Path):
    files = sorted(source.rglob("*"))
    if source.is_symlink() or any(path.is_symlink() for path in files):
        raise ValueError("Artifact export forbids symlinks")
    for path in files:
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
