"""Versioned, deterministic role-balanced league policy (no neural imports)."""

import hashlib
import json
import random
from pathlib import Path
from records import save
from transport import ENGINE, ROOT
from referee import REFEREE_NAME

VERSION = "skallagrim-league-v1"
RULES = "copenhagen-static-v3"
FAILURES = {"timeout", "engine-failure", "illegal-move", "rules-incompatibility"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checkpoint(path, bot):
    path = Path(path).resolve()
    return dict(bot=bot, model=pin(path), metadata=pin(path.with_suffix(".meta.json")))


def position_id(position):
    return digest(
        {k: position[k] for k in ["rules", "start", "moves", "fen", "provenance"]}
    )


def load_positions(path):
    suite = json.loads(Path(path).read_text(encoding="utf8"))
    if suite.get("format") != "skallagrim-league-positions-v1":
        raise ValueError("Unsupported league position suite")
    seen_ids, seen_fens = set(), set()
    for partition in ["training", "evaluation"]:
        positions = suite.get(partition, [])
        if len(positions) < 25:
            raise ValueError("Need at least 25 distinct positions in each partition")
        for position in positions:
            if position.get("rules") != RULES or position.get("id") != position_id(
                position
            ):
                raise ValueError("Changed position identity or rules")
            if position["id"] in seen_ids or position["fen"] in seen_fens:
                raise ValueError("Training/evaluation positions overlap or repeat")
            seen_ids.add(position["id"])
            seen_fens.add(position["fen"])
    return suite


def make_schedule(current, champion, history, positions, games, seed):
    if games < 2 or games % 2:
        raise ValueError("League games must be even for role-swapped pairs")
    rng = random.Random(seed)
    categories = [("self", 20)]
    if champion:
        categories.append(("champion", 50))
    if history:
        categories.append(("historical", 30))
    rows = []
    for pair in range(games // 2):
        category = rng.choices([v[0] for v in categories], [v[1] for v in categories])[
            0
        ]
        enemy = (
            current
            if category == "self"
            else champion
            if category == "champion"
            else rng.choice(history)
        )
        position = rng.choice(positions) if pair % 5 == 4 else None
        for side in ["Attacker", "Defender"]:
            rows.append(
                dict(
                    pair=pair,
                    candidate_side=side,
                    opponent=enemy,
                    category=category,
                    position=position,
                )
            )
    return rows


def manifest(
    path,
    current,
    champion,
    history,
    suite_path,
    games,
    max_attempts,
    seed,
    movetime,
    max_plies,
):
    suite = load_positions(suite_path)
    referee_path = (
        ROOT / REFEREE_NAME
        if (ROOT / REFEREE_NAME).is_file()
        else ROOT / "target/release" / REFEREE_NAME
    )
    import os

    referee_path = Path(os.environ.get("SKALLAGRIM_REFEREE", referee_path))
    value = dict(
        format=VERSION,
        rules=RULES,
        engine=pin(ENGINE),
        referee=pin(referee_path),
        current=current,
        champion=champion,
        history=history,
        positions=pin(suite_path),
        games=games,
        max_attempts=max_attempts,
        seed=seed,
        movetime=movetime,
        max_plies=max_plies,
        schedule=make_schedule(
            current, champion, history, suite["training"], max_attempts, seed
        ),
    )
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(
            "League resume rejected: changed artifacts, positions, schedule, or settings"
        )
    save(path, value)
    return value, suite


def verify_manifest(value):
    if value.get("format") != VERSION or value.get("rules") != RULES:
        raise ValueError("Unsupported league manifest")
    artifacts = [value["engine"], value["referee"], value["positions"]]
    for entry in [value["current"], value.get("champion"), *value["history"]]:
        if entry and entry.get("model"):
            artifacts.extend([entry["model"], entry["metadata"]])
    for artifact in artifacts:
        if pin(artifact["path"]) != artifact:
            raise ValueError("League artifact changed")
    suite = load_positions(value["positions"]["path"])
    expected = make_schedule(
        value["current"],
        value.get("champion"),
        value["history"],
        suite["training"],
        value["max_attempts"],
        value["seed"],
    )
    if value["schedule"] != expected:
        raise ValueError("League schedule changed")
    return suite


def passes(summary, confidence=False):
    if summary.get("games") != 50 or not summary.get("completed"):
        return False
    if any(summary.get("termination_counts", {}).get(reason, 0) for reason in FAILURES):
        return False
    sides = summary.get("by_side", {})
    if any(
        sides.get(side, {}).get("games") != 25
        or (sides.get(side, {}).get("score_rate") or 0) < 0.5
        for side in ["Attacker", "Defender"]
    ):
        return False
    interval = summary.get("paired_bootstrap_95")
    return not confidence or bool(interval and interval[0] > 0.5)


def gate_accepted(summaries, champion_keys):
    return bool(summaries) and all(
        passes(summary, key in champion_keys) for key, summary in summaries.items()
    )


def run_gates(folder, kind, model, champion, historical, suite_path, run, budget):
    from transport import ROOT

    folder = Path(folder)
    suite = load_positions(suite_path)
    hard_openings = folder / "held-out-openings.json"
    save(
        hard_openings,
        dict(format="skallagrim-openings-v1", openings=suite["evaluation"][:25]),
    )
    templates = json.loads((ROOT / "benchmarks/engines.json").read_text())["engines"]
    summaries, confidence_keys = {}, set()
    bots = ["rho"] if kind == "cnn" else ["iota", "beta"]
    opponents = [
        ("champion", champion),
        *[(f"historical-{i}", path) for i, path in enumerate(historical[-3:])],
    ]
    for bot in bots:
        engines = {}
        for name, path in [("candidate", model), *opponents]:
            entry = json.loads(json.dumps(templates[bot]))
            path = Path(path).resolve()
            meta = path.with_suffix(".meta.json")
            entry["options"]["CNNFile" if kind == "cnn" else "NNUEFile"] = str(path)
            entry["artifacts"] = [str(ENGINE.resolve()), str(path), str(meta)]
            entry["command"] = [str(ENGINE.resolve())]
            entry["training_metadata"] = [str(meta)]
            engines[name] = entry
        config = folder / f"{bot}-gate-config.json"
        save(config, dict(engines=engines))
        run(
            "python/benchmark.py",
            "--config",
            config,
            "--candidate",
            "candidate",
            "--opponents",
            *[name for name, _ in opponents],
            "--output",
            folder / f"gate-{bot}",
            "--resume",
            budget=budget,
        )
        for name, _ in opponents:
            key = f"{bot}/{name}"
            summaries[key] = json.loads(
                (folder / f"gate-{bot}" / f"{name}-summary.json").read_text()
            )
            if name == "champion":
                confidence_keys.add(key)
        run(
            "python/benchmark.py",
            "--config",
            config,
            "--candidate",
            "candidate",
            "--opponents",
            "champion",
            "--openings",
            hard_openings,
            "--output",
            folder / f"gate-{bot}-held-out",
            "--resume",
            budget=budget,
        )
        summaries[f"{bot}/held-out"] = json.loads(
            (folder / f"gate-{bot}-held-out/champion-summary.json").read_text()
        )
    return gate_accepted(summaries, confidence_keys), summaries
