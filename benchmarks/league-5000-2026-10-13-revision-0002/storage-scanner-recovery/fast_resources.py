"""A storage budget scoped to one explicitly selected learning run."""

import os
import shutil
from pathlib import Path


class StorageBudget:
    def __init__(self, root, max_bytes, reserve_bytes=2 * 1024**3):
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes
        self.reserve_bytes = reserve_bytes

    def available_bytes(self):
        used = 0
        # DirEntry carries Windows enumeration metadata. Path.rglob followed by
        # is_file/stat repeatedly enumerates directories and requests metadata
        # again for every file. Visit each directory once per exact check.
        directories = [self.root]
        while directories:
            try:
                with os.scandir(directories.pop()) as entries:
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                directories.append(entry.path)
                            elif entry.is_file():
                                used += entry.stat().st_size
                        except FileNotFoundError:
                            pass  # Concurrent atomic replacement or removal.
            except FileNotFoundError:
                pass  # A child may have removed a temporary directory.
        return min(
            self.max_bytes - used,
            shutil.disk_usage(self.root).free - self.reserve_bytes,
        )

    def check(self):
        if self.available_bytes() < 0:
            raise RuntimeError(
                f"Learning-run storage budget exhausted: {self.root}. Completed checkpoints are preserved."
            )
