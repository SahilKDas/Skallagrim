"""Pure helpers shared by NNUE training, diagnostics and bounded experiments."""

import hashlib
import json
import math
from pathlib import Path

import torch

from models import encode


def transform_fen(fen, symmetry):
    if symmetry not in range(8):
        raise ValueError("symmetry must be in 0..7")
    board = encode(fen)[0][:3]
    board = torch.rot90(board, symmetry % 4, dims=(1, 2))
    if symmetry >= 4:
        board = board.flip(2)
    rows = []
    for y in range(11):
        row, empty = "", 0
        for x in range(11):
            piece = next(("adk"[i] for i in range(3) if board[i, y, x]), None)
            if piece is None:
                empty += 1
            else:
                row += (str(empty) if empty else "") + piece
                empty = 0
        rows.append(row + (str(empty) if empty else ""))
    return "/".join(rows) + " " + fen.split()[1]


def augment_row(row, symmetry):
    return row | {
        "fen": transform_fen(row["fen"], symmetry),
        "history": [transform_fen(fen, symmetry) for fen in row["history"]],
    }


def sample_key(row):
    value = [row["game"], row["fen"], row["history"]]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def validate_labels(manifest, data_hash, training_games):
    if (
        manifest.get("data_sha256") != data_hash
        or manifest.get("perspective") != "Attacker"
        or manifest.get("training_only") is not True
    ):
        raise ValueError(
            "Search labels require matching data, attacker perspective and training-only provenance"
        )
    for name in ["engine_sha256", "model_sha256"]:
        value = manifest.get(name, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Search label artifact provenance missing")
        artifact = Path(manifest.get(name.replace("_sha256", "_path"), ""))
        if (
            not artifact.is_file()
            or hashlib.sha256(artifact.read_bytes()).hexdigest() != value
        ):
            raise ValueError("Search label artifact hash mismatch")
    for record in manifest["labels"].values():
        if (
            record["game"] not in training_games
            or not math.isfinite(record["value"])
            or not -1 <= record["value"] <= 1
        ):
            raise ValueError("Invalid or non-training search label")
    return manifest["labels"]


def target_for(row, labels, mix):
    if not mix:
        return row["target"]
    record = labels.get(sample_key(row))
    if record is None or record["game"] != row["game"]:
        raise ValueError("Missing search label; refusing fabricated target")
    return (1 - mix) * row["target"] + mix * record["value"]


def training_offsets(index, valid, cap, rng):
    result = []
    for game in sorted(index.groups):
        if game not in valid:
            offsets = index.groups[game]
            result.extend(
                rng.sample(offsets, cap) if cap and len(offsets) > cap else offsets
            )
    return sorted(result)
