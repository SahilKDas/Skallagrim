"""Iterative self-play/replay/training driver. Every candidate needs 1,000 games.

Runs synchronously; each generation has durable files and metadata. An initial
baseline warm start is followed by neural self-play, never called proven strength.
"""

import argparse
import json
import subprocess
import sys
import os
import signal
import hashlib
from pathlib import Path
from records import save
from transport import ROOT
from dataset import make_replay, archive_data
from resources import StorageBudget


def validate_checkpoint(path):
    path = Path(path)
    for artifact in [path, path.with_suffix(".pt"), path.with_suffix(".meta.json")]:
        if not artifact.is_file():
            raise FileNotFoundError(f"Checkpoint artifact is missing: {artifact}")
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    if metadata.get("smoke") or metadata.get("training_games", 0) < 1000:
        raise ValueError("Checkpoint requires at least 1,000 completed training games")
    for name, expected in metadata.get("model_artifacts", {}).items():
        artifact = path.parent / name
        if artifact.parent.resolve() != path.parent.resolve():
            raise ValueError("Checkpoint artifact path escapes its directory")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint artifact changed: {artifact}")
    return metadata


def run(*args, budget=None):
    if budget:
        budget.check()
    command = [sys.executable, *map(str, args)]
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        while True:
            try:
                code = process.wait(timeout=0.5)
                if code:
                    raise subprocess.CalledProcessError(code, command)
                if budget:
                    budget.check()
                return
            except subprocess.TimeoutExpired:
                if budget:
                    budget.check()
    except BaseException:
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            else:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            process.wait(timeout=5)
        raise


def main():
    from league import (
        checkpoint as league_checkpoint,
        manifest as league_manifest,
        load_positions,
        pin,
        verify_manifest,
        run_gates,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["cnn", "nnue"], required=True)
    p.add_argument(
        "--league-positions",
        type=Path,
        default=ROOT / "benchmarks/league-positions.json",
    )
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--games", type=int, default=1250)
    p.add_argument("--movetime", type=int, default=100)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--window-games", type=int, default=20000)
    p.add_argument("--replay-max-gib", type=float, default=8)
    p.add_argument("--max-storage-gib", type=float, default=16)
    p.add_argument(
        "--retain-raw",
        action="store_true",
        help="Keep completed JSONL files instead of verified lossless gzip archives",
    )
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if args.generations <= 0 or args.movetime <= 0 or args.epochs <= 0:
        raise SystemExit("generations, movetime and epochs must be positive")
    if args.games < 1250 or args.window_games < 1250:
        raise SystemExit("Use >=1,250 games: at least 1,000 training plus validation.")
    if args.replay_max_gib <= 0:
        raise SystemExit("--replay-max-gib must be positive")
    if args.max_storage_gib <= 0:
        raise SystemExit("--max-storage-gib must be positive")
    args.run = args.run.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    budget = StorageBudget(args.run, int(args.max_storage_gib * 1024**3))
    if args.games % 2:
        raise SystemExit("Use an even game count for role-balanced pairs")
    if not args.league_positions.is_file():
        run(
            "python/make_league_positions.py",
            "--output",
            args.league_positions,
            budget=budget,
        )
    position_suite = load_positions(args.league_positions)
    config_path = args.run / "config.json"
    config = vars(args) | {
        "run": str(args.run),
        "league_positions": pin(args.league_positions),
        "league_version": "skallagrim-league-v1",
    }
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        for key in [
            "kind",
            "games",
            "movetime",
            "epochs",
            "window_games",
            "device",
            "league_positions",
            "league_version",
        ]:
            if previous.get(key) != config[key]:
                raise SystemExit(
                    f"Resume rejected: changed {key}; use a fresh --run directory"
                )
    save(config_path, config)
    checkpoint = None
    promoted_history = []
    for generation in range(args.generations):
        folder = args.run / f"generation-{generation:04d}"
        folder.mkdir(exist_ok=True)
        samples = folder / "selfplay.jsonl"
        model = folder / f"{args.kind}.json"
        # Validated completed generations can be resumed without retraining.
        if (folder / "complete.json").exists():
            complete = json.loads((folder / "complete.json").read_text())
            verify_manifest(json.loads((folder / "league.json").read_text()))
            validate_checkpoint(Path(complete["model"]))
            checkpoint = Path(complete["promoted_model"])
            validate_checkpoint(checkpoint)
            if complete["accepted"]:
                promoted_history.append(checkpoint)
            if not args.retain_raw:
                archive_data(samples, budget.check)
                archive_data(folder / "replay.jsonl", budget.check)
            continue
        if not model.with_suffix(".meta.json").exists():
            bot = (
                "Baseline"
                if checkpoint is None
                else "Rho"
                if args.kind == "cnn"
                else "Beta"
            )
            cmd = [
                "python/selfplay.py",
                "--games",
                args.games,
                "--movetime",
                args.movetime,
                "--seed",
                10000 + generation,
                "--bot",
                bot,
                "--output",
                samples,
            ]
            if checkpoint:
                cmd += ["--cnn" if args.kind == "cnn" else "--nnue", checkpoint]
            current = (
                league_checkpoint(checkpoint, bot)
                if checkpoint
                else dict(bot="Baseline", model=None)
            )
            history = [
                league_checkpoint(path, bot) for path in promoted_history[:-1][-8:]
            ]
            league_manifest(
                folder / "league.json",
                current,
                current if checkpoint else None,
                history,
                args.league_positions,
                args.games,
                args.games * 10,
                10000 + generation,
                args.movetime,
                600,
            )
            cmd += ["--league", folder / "league.json"]
            run(*cmd, budget=budget)
            replay = folder / "replay.jsonl"
            sources = [
                s
                for s in sorted(args.run.glob("generation-*/selfplay.jsonl*"))
                if s.name in {"selfplay.jsonl", "selfplay.jsonl.gz"}
                and s.parent.name <= folder.name
                and not (s.suffix == ".gz" and s.with_suffix("").exists())
            ]
            selection = make_replay(
                sources,
                replay,
                args.window_games,
                min(int(args.replay_max_gib * 1024**3), budget.available_bytes()),
                check=budget.check,
                excluded_fens={row["fen"] for row in position_suite["evaluation"]},
            )
            save(folder / "replay-selection.json", selection)
            cmd = [
                "python/train.py",
                "--data",
                replay,
                "--kind",
                args.kind,
                "--output",
                model,
                "--epochs",
                args.epochs,
                "--device",
                args.device,
            ]
            if checkpoint:
                cmd += ["--resume", checkpoint.with_suffix(".pt")]
            run(*cmd, budget=budget)
        verify_manifest(json.loads((folder / "league.json").read_text()))
        accepted = True
        validate_checkpoint(model)
        summary = None
        if checkpoint is not None:
            accepted, summary = run_gates(
                folder,
                args.kind,
                model,
                checkpoint,
                promoted_history[:-1],
                args.league_positions,
                run,
                budget,
            )
        promoted = model if accepted else checkpoint
        save(
            folder / "complete.json",
            dict(
                generation=generation,
                training_games=json.loads(model.with_suffix(".meta.json").read_text())[
                    "training_games"
                ],
                model=str(model),
                accepted=accepted,
                promoted_model=str(promoted),
                gate=summary,
                promotion="role-balanced league: 50-game/300-ms gauntlets per opponent and held-out suite"
                if checkpoint
                else "initial bootstrap; external strength unproven",
            ),
        )
        checkpoint = promoted
        if accepted:
            promoted_history.append(model)
        if not args.retain_raw:
            archive_data(samples, budget.check)
            archive_data(folder / "replay.jsonl", budget.check)


if __name__ == "__main__":
    main()
