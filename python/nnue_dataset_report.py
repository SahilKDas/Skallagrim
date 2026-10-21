"""Stream coverage diagnostics without altering audited training records."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch

from dataset import DatasetIndex, open_data
from models import encode_nnue
from records import save
from transport import pin_current_process


def report(data, split_path):
    index = DatasetIndex(data)
    split = json.loads(Path(split_path).read_text())
    train, valid = set(split["training"]), set(split["validation"])
    if (
        split["data_sha256"] != index.sha256
        or train & valid
        or train | valid != set(index.groups)
    ):
        raise ValueError("Changed, overlapping or incomplete split")
    sides, roles, lengths, openings, endings, tactics = (Counter() for _ in range(6))
    games_seen = set()
    game_plies = {}
    positions = 0
    with open_data(data) as stream:
        while line := stream.readline():
            row = json.loads(line)
            game = row.get("game")
            if game not in index.groups or row.get("target") is None:
                continue
            partition = "training" if game in train else "validation"
            game_plies[game] = max(
                game_plies.get(game, 0), row.get("ply", len(row["history"]) - 1) + 1
            )
            positions += 1
            sides[f"{partition}/{row['fen'].split()[1]}"] += 1
            features = encode_nnue(row["fen"], row["history"], "v2")
            if features[364 + 29].item() == 0:
                tactics[f"{partition}/clear-corner-ray"] += 1
            if features[364 + 24].item() == 0:
                tactics[f"{partition}/adjacent-attacker"] += 1
            if features[366].item():
                tactics[f"{partition}/repetition"] += 1
            if game not in games_seen:
                games_seen.add(game)
                roles[f"{partition}/{index.roles.get(game, 'unknown')}"] += 1
                lengths[f"{partition}/{len(index.groups[game])}"] += 1
                opening = (row.get("league") or {}).get(
                    "position_id"
                ) or hashlib.sha256(json.dumps(row["history"][:5]).encode()).hexdigest()
                openings[f"{partition}/{opening}"] += 1
                endings[
                    f"{partition}/{(row.get('outcome') or {}).get('reason', 'unknown')}"
                ] += 1
    played_plies = Counter(
        f"{'training' if g in train else 'validation'}/{n}"
        for g, n in game_plies.items()
    )
    return dict(
        data_sha256=index.sha256,
        games=len(games_seen),
        positions=positions,
        roles=dict(roles),
        sample_sides=dict(sides),
        samples_per_game_histogram=dict(lengths),
        played_plies_histogram=dict(played_plies),
        opening_history_prefixes=dict(openings),
        endings=dict(endings),
        tactical_proxies=dict(tactics),
        limitations=[
            "Position proxies are not solved tactical labels",
            "Samples per game may omit opening plies",
            "Fort, shieldwall and capture coverage require replay-event classification; unknown is not absent",
        ],
        future_collection_gaps=[
            "Diverse attacker starts",
            "King escape and capture decisions",
            "Shieldwall opportunities",
            "Independently verified forts",
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    pin_current_process()
    torch.set_num_threads(1)
    save(args.output, report(args.data, args.split))


if __name__ == "__main__":
    main()
