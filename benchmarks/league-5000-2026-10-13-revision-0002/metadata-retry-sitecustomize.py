"""Versioned metadata-only recovery; leaves frozen computational inputs intact."""
import hashlib
import os
from pathlib import Path
import time

expected = os.environ.get("SKALLAGRIM_METADATA_RECOVERY_SHA")
root_name = os.environ.get("SKALLAGRIM_METADATA_RECOVERY_ROOT")
if expected and root_name:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expected:
        os._exit(78)
    root = Path(root_name).resolve()
    original_replace = os.replace
    names = {"collection-progress.json", "status.json", "active-stage.json", "published.json", "config.json", "complete.json", "artifact-index.json"}
    def replace_metadata(source, destination, *args, **kwargs):
        target = Path(destination).resolve()
        protected = root / "runtime"
        eligible = target.is_relative_to(root) and not target.is_relative_to(protected) and (target.name in names or target.name.endswith(".progress.json"))
        if not eligible:
            return original_replace(source, destination, *args, **kwargs)
        for attempt in range(9):
            try:
                return original_replace(source, destination, *args, **kwargs)
            except PermissionError as error:
                if getattr(error, "winerror", None) not in {5,32,33} or attempt == 8:
                    raise
                time.sleep(min(0.01 * 2**attempt, 0.2))
    os.replace = replace_metadata
