"""Generate authoritative, outcome-labelled samples; truncated games stay unlabelled."""

import argparse
import gzip
import json
import random
import contextlib
from pathlib import Path
from transport import Engine
from referee import Referee
from league import verify_manifest, digest, FAILURES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=1250)
    p.add_argument("--movetime", type=int, default=30)
    p.add_argument("--max-plies", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--bot", choices=["Baseline", "Rho", "Iota", "Beta"], default="Baseline"
    )
    p.add_argument("--nnue", type=Path)
    p.add_argument("--cnn", type=Path)
    p.add_argument(
        "--league",
        type=Path,
        help="Pinned league manifest; uses separate engines for each side",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exploration", type=float, default=0.15)
    p.add_argument(
        "--max-attempts",
        type=int,
        help="Stop rather than endlessly retrying truncated games; default 10 times --games",
    )
    args = p.parse_args()
    if args.games <= 0 or args.movetime <= 0 or args.max_plies <= 0:
        raise SystemExit("games, movetime and max-plies must be positive")
    if not 0 <= args.exploration <= 1:
        raise SystemExit("exploration must be between 0 and 1")
    max_attempts = (
        args.max_attempts if args.max_attempts is not None else args.games * 10
    )
    if max_attempts < args.games:
        raise SystemExit("max-attempts must be at least games")
    rng = random.Random(args.seed)
    league = json.loads(args.league.read_text()) if args.league else None
    suite = verify_manifest(league) if league else None
    if league and any(
        league[key] != value
        for key, value in dict(
            games=args.games,
            seed=args.seed,
            movetime=args.movetime,
            max_plies=args.max_plies,
            max_attempts=max_attempts,
        ).items()
    ):
        raise ValueError("Self-play settings do not match league manifest")
    held_out = {row["fen"] for row in suite["evaluation"]} if suite else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    options = {"Bot": args.bot}
    if args.nnue:
        options["NNUEFile"] = str(args.nnue.resolve())
    if args.cnn:
        options["CNNFile"] = str(args.cnn.resolve())
    with (
        contextlib.ExitStack() as processes,
        Referee(
            log_dir=args.output.parent / (args.output.stem + "-referee")
        ) as referee,
        gzip.open(
            args.output.with_suffix(".referee.jsonl.gz"), "wt", encoding="utf8"
        ) as rulings,
        args.output.open("w", encoding="utf8") as out,
    ):
        completed = 0
        completed_by_side = {"Attacker": 0, "Defender": 0}
        game = 0
        pool = {}

        def get_engine(entry, role):
            key = (digest(entry), role)
            if key not in pool:
                opts = {"Bot": entry["bot"]}
                if entry.get("model"):
                    opts["CNNFile" if entry["bot"] == "Rho" else "NNUEFile"] = entry[
                        "model"
                    ]["path"]
                pool[key] = processes.enter_context(Engine(options=opts))
            return pool[key]

        if not league:
            engine = processes.enter_context(Engine(options=options))
        while completed < args.games:
            if game >= max_attempts:
                raise RuntimeError(
                    f"Only {completed}/{args.games} completed after {max_attempts} attempts; partial corpus preserved. Increase --max-attempts or --max-plies."
                )
            plan = league["schedule"][game] if league else None
            if plan and completed_by_side[plan["candidate_side"]] >= args.games // 2:
                game += 1
                continue
            if plan:
                candidate = get_engine(league["current"], "candidate")
                enemy = get_engine(plan["opponent"], "opponent")
                players = {
                    plan["candidate_side"]: candidate,
                    "Defender"
                    if plan["candidate_side"] == "Attacker"
                    else "Attacker": enemy,
                }
            else:
                players = {"Attacker": engine, "Defender": engine}
            position = plan["position"] if plan else None
            start = position["start"] if position else None
            history = list(position["moves"]) if position else []
            for player in set(players.values()):
                player.send("ucinewgame")
                player.ready()
                player.position(history, start)
            samples = []
            state = referee.position(history, start)
            contaminated = state["fen"] in held_out
            failure = None
            for ply in range(args.max_plies):
                if state["outcome"]:
                    break
                referee.start_clock(args.movetime + 1000)
                engine = players[state["side"]]
                try:
                    best, elapsed, policy, _ = engine.go(args.movetime)
                    if best not in state["moves"]:
                        raise ValueError("self-play returned illegal move")
                except (TimeoutError, RuntimeError, ValueError, OSError) as exc:
                    if not league:
                        raise
                    failure = (
                        "timeout"
                        if isinstance(exc, TimeoutError)
                        else "illegal-move"
                        if isinstance(exc, ValueError)
                        else "engine-failure"
                    )
                    if not referee.snapshot()["outcome"]:
                        referee.call("forfeit", reason=failure, evidence=str(exc))
                    state = referee.snapshot()
                    break
                move = (
                    rng.choice(state["moves"])
                    if rng.random() < args.exploration
                    else best
                )
                if not plan or state["side"] == plan["candidate_side"]:
                    samples.append(
                        dict(
                            format="skallagrim-sample-v1",
                            game=f"{args.seed}:{game}",
                            fen=state["fen"],
                            history=state["history"],
                            legal=state["moves"],
                            policy=policy or [[best, 1.0]],
                            played_move=move,
                            bot=league["current"]["bot"] if league else args.bot,
                            rules=state["rules"],
                            ply=ply,
                            league=dict(
                                manifest=digest(league),
                                pair=plan["pair"],
                                candidate_side=plan["candidate_side"],
                                opponent=plan["opponent"],
                                category=plan["category"],
                                position_id=position["id"] if position else "standard",
                                seed=args.seed,
                            )
                            if plan
                            else None,
                        )
                    )
                state = referee.play(move)
                history.append(move)
                contaminated |= state["fen"] in held_out
                for player in set(players.values()):
                    mirror = player.play(move)
                    if (
                        mirror["fen"] != state["fen"]
                        or mirror["outcome"] != state["outcome"]
                    ):
                        raise RuntimeError(
                            "search engine disagrees with authoritative referee"
                        )
            rulings.write(
                json.dumps(
                    {"game": f"{args.seed}:{game}", "league": plan, **referee.export()}
                )
                + "\n"
            )
            rulings.flush()
            outcome = state["outcome"]
            target = (
                None
                if outcome is None
                or failure
                or contaminated
                or outcome["reason"] in FAILURES
                else 1.0
                if outcome["winner"] == "Attacker"
                else -1.0
                if outcome["winner"] == "Defender"
                else 0.0
            )
            for row in samples:
                row.update(
                    target=target,
                    outcome=outcome,
                    adjudication="held-out-contamination"
                    if contaminated
                    else "infrastructure-failure"
                    if failure
                    else "terminal"
                    if outcome
                    else "truncated",
                )
                out.write(json.dumps(row) + "\n")
            completed += target is not None and bool(samples)
            if plan and target is not None and samples:
                completed_by_side[plan["candidate_side"]] += 1
            if failure:
                for player in set(pool.values()):
                    player.close()
                pool.clear()
            out.flush()
            print(
                json.dumps(
                    dict(
                        game=game,
                        completed=completed,
                        plies=len(samples),
                        outcome=outcome,
                    )
                ),
                flush=True,
            )
            game += 1


if __name__ == "__main__":
    main()
