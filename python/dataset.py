"""Streaming JSONL indices and whole-game replay assembly (plain or gzip)."""

import gzip
import hashlib
import json
from pathlib import Path


def open_data(path):
    path = Path(path)
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


class DatasetIndex:
    def __init__(self, path, excluded_fens=()):
        self.path = Path(path)
        self.groups = {}
        self.sides = {}
        self.sizes = {}
        digest = hashlib.sha256()
        excluded_fens = set(excluded_fens)
        contaminated = set()
        with open_data(path) as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("fen") in excluded_fens:
                    contaminated.add(row["game"])
                if row.get("target") is None:
                    continue
                game = row["game"]
                self.groups.setdefault(game, []).append(offset)
                self.sizes[game] = self.sizes.get(game, 0) + len(line)
                counts = self.sides.setdefault(game, {"a": 0, "d": 0})
                side = row.get("fen", "").rsplit(" ", 1)[-1]
                if side in counts:
                    counts[side] += 1
        self.sha256 = digest.hexdigest()
        for game in contaminated:
            self.groups.pop(game, None)
            self.sides.pop(game, None)
            self.sizes.pop(game, None)

    @staticmethod
    def read_batch(stream, offsets):
        rows = []
        for offset in offsets:
            stream.seek(offset)
            rows.append(json.loads(stream.readline()))
        return rows


def make_replay(sources, output, max_games, max_bytes, minimum_games=1250, check=None, excluded_fens=()):
    selected = []
    seen = set()
    total = 0
    full = False
    for source in reversed(sources):
        index = DatasetIndex(source, excluded_fens)
        chosen = []
        for game in reversed(list(index.groups)):
            if game in seen:
                continue
            size = index.sizes[game]
            if len(seen) >= max_games or total + size > max_bytes:
                full = True
                break
            seen.add(game)
            total += size
            chosen.append((game, index.groups[game]))
        if chosen:
            selected.append((Path(source), sorted(chosen, key=lambda item: item[1][0])))
        if full or len(seen) >= max_games:
            break
    if len(seen) < minimum_games:
        raise ValueError(
            f"Replay budget fits only {len(seen)} completed games; need {minimum_games}. Increase --replay-max-gib."
        )
    output = Path(output)
    staging = output.with_suffix(output.suffix + ".tmp")
    with staging.open("wb") as dest:
        checked = 0
        for source, games in selected:
            with open_data(source) as stream:
                for _, offsets in games:
                    for offset in offsets:
                        stream.seek(offset)
                        dest.write(stream.readline())
                        if check and dest.tell() - checked >= 32 * 1024**2:
                            check()
                            checked = dest.tell()
    if check:
        check()
    staging.replace(output)
    return {
        "games": len(seen),
        "bytes": total,
        "sources": [
            {"path": str(source), "games": [game for game, _ in games]}
            for source, games in selected
        ],
    }


def archive_data(path, check=None):
    """Replace our completed corpus with a verified lossless gzip archive."""
    path = Path(path)
    if not path.exists():
        return
    archive = path.with_suffix(path.suffix + ".gz")
    staging = archive.with_suffix(archive.suffix + ".tmp")
    with path.open("rb") as src, staging.open("wb") as dest:
        with gzip.GzipFile(fileobj=dest, mode="wb", compresslevel=3, mtime=0) as zipped:
            while chunk := src.read(1024 * 1024):
                zipped.write(chunk)
                if check:
                    check()
    with path.open("rb") as src, gzip.open(staging, "rb") as restored:
        if (
            hashlib.file_digest(src, "sha256").digest()
            != hashlib.file_digest(restored, "sha256").digest()
        ):
            raise ValueError(f"Archive verification failed: {path}")
    staging.replace(archive)
    path.unlink()
