"""Streaming JSONL indices and whole-game replay assembly (plain or gzip)."""

import gzip
import hashlib
import json
import sqlite3
import zlib
from pathlib import Path


def open_data(path):
    path = Path(path)
    if path.suffix == ".sqlite":
        return SqliteStream(path)
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


class SqliteStream:
    """Opaque offsets are row IDs; preserves read_batch's seek/read contract."""

    def __init__(self, path):
        self.connection = sqlite3.connect(
            Path(path).resolve().as_uri() + "?mode=ro", uri=True
        )
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version != 1:
            self.connection.close()
            raise ValueError("Unsupported compressed replay version")
        self.position = 1

    def tell(self):
        return self.position

    def seek(self, position):
        self.position = position

    def readline(self):
        value = self.connection.execute(
            "SELECT body FROM sample WHERE id=?", (self.position,)
        ).fetchone()
        if value is None:
            return b""
        self.position += 1
        return zlib.decompress(value[0])

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def make_replay_sqlite(
    sources, output, max_games, max_bytes, minimum_games, check=None, excluded_fens=()
):
    selected, seen = [], set()
    for source in reversed(sources):
        index = DatasetIndex(source, excluded_fens)
        games = []
        for game in reversed(list(index.groups)):
            if game not in seen and len(seen) < max_games:
                seen.add(game)
                games.append(game)
        if games:
            selected.append((Path(source), set(games)))
        if len(seen) >= max_games:
            break
    if len(seen) < minimum_games:
        raise ValueError("Compressed replay has insufficient completed games")
    output = Path(output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    raw_bytes, count = 0, 0
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA user_version=1")
        connection.execute(
            "CREATE TABLE sample(id INTEGER PRIMARY KEY, body BLOB NOT NULL)"
        )
        for source, games in reversed(selected):
            with open_data(source) as stream:
                while line := stream.readline():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row["game"] not in games or row.get("target") is None:
                        continue
                    connection.execute(
                        "INSERT INTO sample(body) VALUES (?)", (zlib.compress(line, 6),)
                    )
                    count += 1
                    raw_bytes += len(line)
                    if count % 1000 == 0:
                        connection.commit()
                        if temporary.stat().st_size > max_bytes:
                            raise ValueError(
                                "Compressed replay exceeds disk cap; completed data retained"
                            )
                        if check:
                            check()
        connection.commit()
        if temporary.stat().st_size > max_bytes:
            raise ValueError("Compressed replay exceeds disk cap")
        if check:
            check()
    finally:
        connection.close()
    temporary.replace(output)
    return dict(
        format="skallagrim-replay-sqlite-v1",
        games=len(seen),
        samples=count,
        raw_bytes=raw_bytes,
        bytes=output.stat().st_size,
        sources=[
            dict(path=str(source), games=sorted(games)) for source, games in selected
        ],
    )


class DatasetIndex:
    def __init__(self, path, excluded_fens=()):
        self.path = Path(path)
        self.groups = {}
        self.sides = {}
        self.sizes = {}
        self.roles = {}
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
                role = (row.get("league") or {}).get("candidate_side")
                if role:
                    if game in self.roles and self.roles[game] != role:
                        raise ValueError("Candidate role changed within a game")
                    self.roles[game] = role
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
            self.roles.pop(game, None)

    @staticmethod
    def read_batch(stream, offsets):
        rows = []
        for offset in offsets:
            stream.seek(offset)
            rows.append(json.loads(stream.readline()))
        return rows


def make_replay(
    sources,
    output,
    max_games,
    max_bytes,
    minimum_games=1250,
    check=None,
    excluded_fens=(),
):
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
