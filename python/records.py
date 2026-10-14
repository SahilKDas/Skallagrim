"""Atomic JSON records without importing a neural runtime."""

import json
import os
import tempfile
import time
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as out:
            temporary = Path(out.name)
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        # Windows readers/scanners can briefly deny delete-sharing. Retry the
        # atomic replacement, never fall back to truncating the committed file.
        for attempt in range(9):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 8:
                    raise
                time.sleep(min(0.01 * 2**attempt, 0.2))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
