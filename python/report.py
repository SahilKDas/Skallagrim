"""Summarize completed paired gauntlets without turning forfeits into strength proof."""

import argparse

import json

from pathlib import Path

from records import save


FAILURES = {"timeout", "engine-failure", "illegal-move", "rules-incompatibility"}

EXPECTED = {
    (c, o)
    for c in ["rho", "iota", "beta"]
    for o in ["taflzero", "opentafl", "luizz", "nicholas"]
} | {("rho", "iota"), ("rho", "beta"), ("iota", "beta")}


def main():

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args()

    rows = []
    complete = set()

    for path in sorted(args.root.rglob("*-summary.json")):
        summary = json.loads(path.read_text())
        manifest = json.loads((path.parent / "manifest.json").read_text())

        candidate = manifest["options"]["candidate"]
        opponent = path.name.removesuffix("-summary.json")

        failure_count = sum(
            summary["termination_counts"].get(key, 0) for key in FAILURES
        )

        recorded_games = [
            json.loads(game.read_text())
            for game in path.parent.glob(opponent + "-??.json")
        ]
        played = [game for game in recorded_games if game["reason"] not in FAILURES]
        played_by_side = {}
        for side in ["Attacker", "Defender"]:
            values = [
                game["score"] for game in played if game["candidate_side"] == side
            ]
            played_by_side[side] = dict(
                games=len(values),
                wins=values.count(1),
                draws=values.count(0.5),
                losses=values.count(0),
                score_rate=sum(values) / len(values) if values else None,
            )
        record = dict(
            candidate=candidate,
            opponent=opponent,
            summary=summary,
            failures=failure_count,
            played_games=len(played),
            played_score=sum(game["score"] for game in played) / len(played)
            if played
            else None,
            played_by_side=played_by_side,
            manifest=str(path.parent / "manifest.json"),
            cpu=manifest["options"].get("cpu"),
            strength_evidence=manifest["strength_evidence"],
            settings={
                key: manifest["options"].get(key)
                for key in ["games", "movetime", "ipc_allowance", "max_plies"]
            },
            platform=manifest.get("platform", "unrecorded"),
            processor=manifest.get("processor", "unrecorded"),
            rules=manifest.get("rules", "unrecorded"),
        )

        rows.append(record)

        if (
            summary["completed"]
            and summary["games"] == 50
            and manifest["options"]["movetime"] == 300
            and len(recorded_games) == 50
            and all(
                summary["by_side"][side]["games"] == 25
                for side in ["Attacker", "Defender"]
            )
            and all(
                sum(game["candidate_side"] == side for game in recorded_games) == 25
                for side in ["Attacker", "Defender"]
            )
        ):
            complete.add((candidate, opponent))

    missing = sorted(EXPECTED - complete)
    complete &= EXPECTED
    configurations = sorted(
        {json.dumps(row["settings"], sort_keys=True) for row in rows}
    )
    settings_text = (
        "; ".join(
            f"{s['games']} games, {s['movetime']} ms search, {s['ipc_allowance']} ms IPC allowance, {s['max_plies']} plies maximum"
            for s in map(json.loads, configurations)
        )
        or "no match settings recorded"
    )
    environments = sorted({f"{row['processor']} / {row['platform']}" for row in rows})
    cpus = sorted({row["cpu"] for row in rows if row["cpu"] is not None})

    if args.require_complete and missing:
        raise SystemExit(f"Incomplete required pairings: {missing}")

    text = [
        "# Skallagrim gauntlet results",
        "",
        f"Completed required pairings: {len(complete)}/15. Required format: 50 games, 300 ms search requests, 25 swapped-role opening pairs. Recorded settings: {settings_text}.",
        "",
        "These are bootstrap experiments, not a claim of federation-certified superiority. Native Python opponents have documented shieldwall mismatches and use our labeled warm-start weights. Interpret timeout/illegal-move/compatibility forfeits separately from play. Static fort detection and referee rules remain subject to federation compatibility certification.",
        "",
        "| Candidate | Opponent | Games | W/D/L | Match score | Played score | Attacker match | Defender match | Paired 95% interval | Forfeits | CPU |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]

    def rate(x):
        return "—" if x is None else f"{100 * x:.1f}%"

    for record in rows:
        s = record["summary"]
        interval = s["paired_bootstrap_95"]
        ci = "—" if interval is None else f"{rate(interval[0])}–{rate(interval[1])}"

        text.append(
            f"| {record['candidate']} | {record['opponent']} | {s['games']} | {s['wins']}/{s['draws']}/{s['losses']} | {rate(s['score_rate'])} | {rate(record['played_score'])} ({record['played_games']} games) | {rate(s['by_side']['Attacker']['score_rate'])} | {rate(s['by_side']['Defender']['score_rate'])} | {ci} | {record['failures']} | {record['cpu']} |"
        )

    text += [
        "",
        "Role scores conditional on completing play without a protocol failure:",
        "",
        "| Candidate | Opponent | Attacker played | Defender played |",
        "| --- | --- | ---: | ---: |",
    ]
    for record in rows:
        roles = record["played_by_side"]
        a, d = roles["Attacker"], roles["Defender"]
        text.append(
            f"| {record['candidate']} | {record['opponent']} | {rate(a['score_rate'])} ({a['games']} games) | {rate(d['score_rate'])} ({d['games']} games) |"
        )
    text += [
        "",
        "Match scores include protocol forfeits; the last column before CPU counts those forfeits. Played score is conditional on a game finishing without a protocol failure and can be selection-biased; it is not an independent strength estimate. Full termination breakdowns and side results are preserved in JSON summaries. Confidence intervals resample opening pairs, not individual games.",
        "",
        "Recorded environments: "
        + ("; ".join(environments) or "none")
        + f". Affinity CPUs: {cpus or 'unrecorded'}. Detailed resource settings and ambient-load notes are in each manifest; timing and hardware are not inferred from this report's host.",
        "",
        "Incomplete pairings: "
        + ("none" if not missing else ", ".join(f"{c}–{o}" for c, o in missing)),
        "",
        "Every pairing has immutable artifact hashes, model provenance, opening history, process stderr and complete move logs under the supplied run root.",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(text) + "\n", encoding="utf8")

    save(
        args.output.with_suffix(".json"),
        dict(
            complete_pairings=len(complete),
            required_pairings=15,
            missing=missing,
            strength_evidence=False,
            rows=rows,
        ),
    )
    print(args.output, len(complete), "completed pairings")


if __name__ == "__main__":
    main()
