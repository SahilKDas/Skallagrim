"""Paired 50-game/300-ms gauntlets with authoritative rules and replayable logs."""

import argparse
import contextlib
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from records import save
from transport import Engine, ROOT
from referee import Referee


def fingerprint(entry):
    files = {}
    for name in entry.get("artifacts", []):
        path = (ROOT / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(config=entry, sha256=files)


def open_engine(entry, stderr):
    command = [str(ROOT / x[2:]) if x.startswith("./") else x for x in entry["command"]]
    options = {
        k: str(ROOT / v[2:]) if isinstance(v, str) and v.startswith("./") else v
        for k, v in entry.get("options", {}).items()
    }
    engine = Engine(
        command, options, ROOT / entry.get("cwd", "."), stderr, cpu=entry.get("cpu")
    )
    if entry.get("warmup", False):
        try:
            engine.position([])
            engine.go(entry.get("warmup_ms", 1000), 5000)
            engine.ready()
        except BaseException:
            engine.close()
            raise
    return engine


def paired_interval(scores, seed=42):
    if not scores or len(scores) % 2:
        return None
    pairs = [scores[i : i + 2] for i in range(0, len(scores), 2)]
    rng = random.Random(seed)
    samples = sorted(
        sum(sum(rng.choice(pairs)) for _ in pairs) / len(scores) for _ in range(5000)
    )
    return [samples[125], samples[4875]]


def summarize(games, seed):
    scores = [g["score"] for g in games]
    by_side = {}
    for side in ["Attacker", "Defender"]:
        values = [g["score"] for g in games if g["candidate_side"] == side]
        by_side[side] = dict(
            games=len(values),
            wins=values.count(1),
            draws=values.count(0.5),
            losses=values.count(0),
            score_rate=statistics.mean(values) if values else None,
        )
    return dict(
        games=len(games),
        wins=scores.count(1),
        draws=scores.count(0.5),
        losses=scores.count(0),
        score_rate=statistics.mean(scores) if scores else None,
        paired_bootstrap_95=paired_interval(scores, seed)
        if len(scores) % 2 == 0
        else None,
        by_side=by_side,
        termination_counts={
            reason: sum(g["reason"] == reason for g in games)
            for reason in sorted({g["reason"] for g in games})
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "benchmarks/engines.json")
    p.add_argument("--candidate", required=True)
    p.add_argument("--opponents", nargs="+", required=True)
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--openings", type=Path, default=ROOT / "benchmarks/openings.json")
    p.add_argument("--movetime", type=int, default=300)
    p.add_argument("--max-plies", type=int, default=600)
    p.add_argument("--ipc-allowance", type=int, default=25)
    p.add_argument("--cpu", type=int, help="Pin both players to the same logical CPU")
    p.add_argument(
        "--load-note", default="Uncontrolled ambient desktop load; see timing logs"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-unaudited", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when engine hashes and match settings still match",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Allow nonstandard short matches; never label as strength evidence",
    )
    args = p.parse_args()
    if args.movetime <= 0 or args.max_plies <= 0 or args.ipc_allowance < 0:
        raise SystemExit(
            "movetime/max-plies must be positive and IPC allowance nonnegative"
        )
    if args.games < 2 or args.games % 2:
        raise SystemExit("Use a positive even game count for paired sides.")
    if not args.smoke and (args.games != 50 or args.movetime != 300):
        raise SystemExit(
            "Strength gauntlets require exactly 50 games and 300 ms/move; use --smoke for integration checks."
        )
    entries = json.loads(args.config.read_text())["engines"]
    if args.cpu is not None:
        for name in [args.candidate, *args.opponents]:
            entries[name]["cpu"] = args.cpu
    suite = json.loads(args.openings.read_text())
    if len(suite["openings"]) < args.games // 2:
        raise SystemExit("Not enough paired openings")
    args.output.mkdir(parents=True, exist_ok=True)
    # Fail before launching a long match if a referenced binary or model is absent.
    pinned = {
        name: fingerprint(entries[name]) for name in [args.candidate, *args.opponents]
    }
    unaudited = [n for n in pinned if not entries[n].get("rules_audited", False)]
    if not args.smoke:
        for name in [args.candidate, *args.opponents]:
            for artifact in entries[name].get("training_metadata", []):
                if artifact:
                    meta = json.loads((ROOT / artifact).read_text())
                    if meta.get("smoke", False) or meta.get("training_games", 0) < 1000:
                        raise SystemExit(
                            f"{name}: neural strength candidates require >=1,000 completed training games plus validation"
                        )
    if unaudited and not args.allow_unaudited:
        raise SystemExit(
            f"Rules compatibility not audited: {unaudited}. Use --allow-unaudited only for exploratory matches."
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    with Referee() as authority:
        rules_version = authority.snapshot()["rules"]
        referee_path = authority.executable.resolve()
        referee_artifact = (
            referee_path.relative_to(ROOT).as_posix()
            if referee_path.is_relative_to(ROOT)
            else referee_path.as_posix()
        )
    metadata = dict(
        format="skallagrim-gauntlet-v1",
        created=time.time(),
        revision=revision,
        dirty=bool(dirty),
        platform=platform.platform(),
        processor=platform.processor(),
        python=sys.version,
        engines=pinned,
        openings=suite,
        options=vars(args)
        | {
            "config": str(args.config),
            "output": str(args.output),
            "openings": str(args.openings),
        },
        strength_evidence=False,
        provisional_gauntlet=not args.smoke and not unaudited,
        rules=rules_version,
        referee=fingerprint({"artifacts": [referee_artifact]}),
        rules_limitations="static forts plus verified conservative dynamic strategies and human adjudication; general qualitative rules require certification",
        timing=f"{args.movetime} ms search request; dedicated referee clock from receipt of clock request through receipt of move; shared IPC allowance logged separately",
    )
    manifest = args.output / "manifest.json"
    if args.resume and manifest.exists():
        previous = json.loads(manifest.read_text())
        for key in ["engines", "openings", "rules", "referee"]:
            if previous[key] != metadata[key]:
                raise SystemExit(f"Resume rejected: changed {key}")
        for key in [
            "games",
            "movetime",
            "max_plies",
            "ipc_allowance",
            "candidate",
            "opponents",
            "seed",
            "cpu",
            "load_note",
        ]:
            if previous["options"][key] != metadata["options"][key]:
                raise SystemExit(f"Resume rejected: changed {key}")
        metadata = previous
    elif manifest.exists():
        raise SystemExit(
            "Output already contains a run; use --resume or a fresh directory"
        )
    else:
        save(manifest, metadata)
    with (
        contextlib.ExitStack() as lifecycle,
        Referee(log_dir=args.output / "referee") as referee,
    ):
        for opponent in args.opponents:
            games = []
            times = []
            cached = {}
            opponent_log = lifecycle.enter_context(
                (args.output / f"{opponent}.stderr.log").open(
                    "a" if args.resume else "w"
                )
            )
            for i in range(args.games):
                saved = args.output / f"{opponent}-{i:02d}.json"
                if args.resume and saved.exists():
                    record = json.loads(saved.read_text())
                    games.append(record)
                    times.extend(m["elapsed_ms"] for m in record["moves"])
                    continue
                candidate_side = "Attacker" if i % 2 == 0 else "Defender"
                opening = suite["openings"][i // 2]
                history = list(opening["moves"])
                referee.position(history, opening.get("start"))
                state = referee.snapshot()
                if state["outcome"]:
                    raise RuntimeError("opening is terminal")
                record = dict(
                    index=i,
                    pair=i // 2,
                    opponent=opponent,
                    candidate=args.candidate,
                    candidate_side=candidate_side,
                    opening=opening,
                    start=opening.get("start", referee.export()["start"]),
                    moves=[],
                    reason=None,
                    winner=None,
                )
                with contextlib.nullcontext(opponent_log) as err:
                    for name in [args.candidate, opponent]:
                        if name not in cached:
                            cached[name] = open_engine(entries[name], err)
                            lifecycle.callback(cached[name].close)
                    with (
                        contextlib.nullcontext(cached[args.candidate]) as candidate,
                        contextlib.nullcontext(cached[opponent]) as enemy,
                    ):
                        for engine in [candidate, enemy]:
                            engine.send("ucinewgame")
                            engine.ready()
                        for ply in range(args.max_plies):
                            if state["outcome"]:
                                break
                            engine = (
                                candidate if state["side"] == candidate_side else enemy
                            )
                            try:
                                engine.position(history, opening.get("start"))
                                referee.start_clock(args.movetime + args.ipc_allowance)
                                move, elapsed, _, lines = engine.go(
                                    args.movetime, args.ipc_allowance
                                )
                                times.append(elapsed)
                                if elapsed > args.movetime + args.ipc_allowance:
                                    raise TimeoutError("late bestmove")
                                prior_side = state["side"]
                                next_state = referee.play(move)
                            except (
                                TimeoutError,
                                RuntimeError,
                                OSError,
                                ValueError,
                            ) as exc:
                                record.update(
                                    reason="rules-incompatibility"
                                    if "upstream rules rejected" in str(exc)
                                    or "upstream rejected move" in str(exc)
                                    else "timeout"
                                    if isinstance(exc, TimeoutError)
                                    or "deadline:" in str(exc)
                                    else "illegal-move"
                                    if isinstance(exc, ValueError)
                                    else "engine-failure",
                                    winner="Defender"
                                    if state["side"] == "Attacker"
                                    else "Attacker",
                                    error=str(exc),
                                )
                                state = referee.snapshot()
                                if not state["outcome"]:
                                    state = referee.call(
                                        "forfeit",
                                        reason=record["reason"],
                                        evidence=str(exc),
                                    )
                                record.update(
                                    reason=state["outcome"]["reason"],
                                    winner=state["outcome"]["winner"],
                                )
                                break
                            record["moves"].append(
                                dict(
                                    move=move,
                                    side=prior_side,
                                    elapsed_ms=elapsed,
                                    over_search_budget_ms=max(
                                        0, elapsed - args.movetime
                                    ),
                                    analysis=lines,
                                )
                            )
                            history.append(move)
                            state = next_state
                if record["reason"] in [
                    "timeout",
                    "engine-failure",
                    "illegal-move",
                    "rules-incompatibility",
                ]:
                    # A timed-out process may still emit its stale bestmove.
                    # Restart both players after failures to preserve pairing.
                    for process in cached.values():
                        process.close()
                    cached.clear()
                if record["reason"] is None:
                    if state["outcome"]:
                        record.update(
                            reason=state["outcome"]["reason"],
                            winner=state["outcome"]["winner"],
                        )
                    else:
                        state = referee.call(
                            "limit", evidence={"max_plies": args.max_plies}
                        )
                        record.update(reason=state["outcome"]["reason"], winner=None)
                record["referee"] = referee.export()["referee"]
                record["final_fen"] = state["fen"]
                record["score"] = (
                    0.5
                    if record["winner"] is None
                    else int(record["winner"] == candidate_side)
                )
                save(args.output / f"{opponent}-{i:02d}.json", record)
                games.append(record)
                print(
                    json.dumps(
                        dict(
                            opponent=opponent,
                            game=i + 1,
                            score=record["score"],
                            reason=record["reason"],
                            plies=len(record["moves"]),
                        )
                    ),
                    flush=True,
                )
                partial = summarize(games, args.seed)
                partial.update(
                    completed=len(games) == args.games,
                    strength_evidence=metadata["strength_evidence"],
                    latency_ms=dict(mean=statistics.mean(times), max=max(times))
                    if times
                    else None,
                )
                save(args.output / f"{opponent}-summary.json", partial)
            for process in cached.values():
                process.close()
            opponent_log.close()
            partial = summarize(games, args.seed)
            partial.update(
                completed=len(games) == args.games,
                strength_evidence=metadata["strength_evidence"],
                latency_ms=dict(mean=statistics.mean(times), max=max(times))
                if times
                else None,
            )
            save(args.output / f"{opponent}-summary.json", partial)


if __name__ == "__main__":
    main()
