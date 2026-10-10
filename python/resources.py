"""A storage budget scoped to one explicitly selected learning run."""

import shutil
from pathlib import Path


class StorageBudget:
    def __init__(self, root, max_bytes, reserve_bytes=2 * 1024**3):
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes
        self.reserve_bytes = reserve_bytes

    def available_bytes(self):
        used = 0
        for path in self.root.rglob("*"):
            try:
                if path.is_file():
                    used += path.stat().st_size
            except FileNotFoundError:
                pass  # A child may have atomically replaced a temporary file.
        return min(
            self.max_bytes - used,
            shutil.disk_usage(self.root).free - self.reserve_bytes,
        )

    def check(self):
        if self.available_bytes() < 0:
            raise RuntimeError(
                f"Learning-run storage budget exhausted: {self.root}. Completed checkpoints are preserved."
            )
