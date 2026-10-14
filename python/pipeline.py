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
import shutil
import time
from pathlib import Path
from records import save
from transport import ROOT
from dataset import make_replay, make_replay_sqlite, archive_data, DatasetIndex
from resources import StorageBudget


def validate_checkpoint(path, minimum_training=1000, minimum_validation=0):
    path = Path(path)
    for artifact in [path, path.with_suffix(".pt"), path.with_suffix(".meta.json")]:
        if not artifact.is_file():
            raise FileNotFoundError(f"Checkpoint artifact is missing: {artifact}")
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    if metadata.get("smoke") or metadata.get("training_games", 0) < minimum_training:
        raise ValueError("Checkpoint requires at least 1,000 completed training games")
    if metadata.get("validation_games", 0) < minimum_validation:
        raise ValueError("Checkpoint validation game requirement not met")
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
                code = process.wait(timeout=5)
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
    p.add_argument("--initial-champion", type=Path)
    p.add_argument("--training-games", type=int, default=1000)
    p.add_argument("--validation-games", type=int, default=250)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--cpu", type=int)
    p.add_argument("--storage-root", type=Path)
    p.add_argument("--collection-cpus", type=int, nargs="+")
    p.add_argument("--collection-source", type=Path)
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
    if (
        args.training_games < 1000
        or args.validation_games <= 0
        or args.games != args.training_games + args.validation_games
        and args.initial_champion
    ):
        raise SystemExit(
            "Seeded campaign counts must match training plus validation requirements"
        )
    if args.initial_champion:
        args.initial_champion = args.initial_champion.resolve()
        validate_checkpoint(args.initial_champion, minimum_validation=1)
    budget = StorageBudget(
        args.storage_root or args.run, int(args.max_storage_gib * 1024**3)
    )
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
        "initial_champion": league_checkpoint(
            args.initial_champion, "Rho" if args.kind == "cnn" else "Beta"
        )
        if args.initial_champion
        else None,
        "storage_root": str(args.storage_root) if args.storage_root else None,
        "collection_source": str(args.collection_source.resolve())
        if args.collection_source
        else None,
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
            "initial_champion",
            "training_games",
            "validation_games",
            "batch",
            "cpu",
            "collection_cpus",
            "collection_source",
        ]:
            if previous.get(key) != config[key]:
                raise SystemExit(
                    f"Resume rejected: changed {key}; use a fresh --run directory"
                )
    save(config_path, config)
    checkpoint = args.initial_champion
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
            if args.collection_source:
                if (
                    generation != 0
                    or not args.collection_cpus
                    or len(args.collection_cpus) < 2
                ):
                    raise ValueError(
                        "Migration requires a first-generation parallel collection"
                    )
                original = json.loads(
                    (args.collection_source / "league.json").read_text()
                )
                verify_manifest(original)
                from transport import ENGINE
                from referee import REFEREE_NAME

                referee_binary = Path(
                    os.environ.get("SKALLAGRIM_REFEREE", ROOT / REFEREE_NAME)
                )
                identities = [
                    (
                        original["current"]["model"]["sha256"],
                        current["model"]["sha256"],
                    ),
                    (
                        original["current"]["metadata"]["sha256"],
                        current["metadata"]["sha256"],
                    ),
                    (original["engine"]["sha256"], pin(ENGINE)["sha256"]),
                    (original["referee"]["sha256"], pin(referee_binary)["sha256"]),
                    (
                        original["positions"]["sha256"],
                        pin(args.league_positions)["sha256"],
                    ),
                ]
                if any(a != b for a, b in identities) or any(
                    original[key] != value
                    for key, value in dict(
                        games=args.games,
                        max_attempts=args.games * 10,
                        seed=10000 + generation,
                        movetime=args.movetime,
                        max_plies=600,
                    ).items()
                ):
                    raise ValueError(
                        "Migration rejected: changed checkpoints, binaries, rules, positions or searches"
                    )
                league_path = folder / "league.json"
                if (
                    league_path.exists()
                    and json.loads(league_path.read_text()) != original
                ):
                    raise ValueError("Migrated schedule changed on resume")
                save(league_path, original)
            else:
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
            if args.initial_champion:
                cmd += [
                    "--resume-collection",
                    "--storage-root",
                    budget.root,
                    "--max-storage-gib",
                    args.max_storage_gib,
                ]
            if args.cpu is not None:
                cmd += ["--cpu", args.cpu]
            if args.collection_cpus and len(args.collection_cpus) > 1:
                cmd = [
                    "python/parallel_collection.py",
                    "--league",
                    folder / "league.json",
                    "--output",
                    samples,
                    "--cpus",
                    *args.collection_cpus,
                    "--storage-root",
                    budget.root,
                    "--max-storage-gib",
                    args.max_storage_gib,
                ]
                if args.collection_source:
                    cmd += ["--source", args.collection_source]
            run(*cmd, budget=budget)
            replay = folder / (
                "replay.sqlite" if args.initial_champion else "replay.jsonl"
            )
            sources = [
                s
                for s in sorted(args.run.glob("generation-*/selfplay.jsonl*"))
                if s.name in {"selfplay.jsonl", "selfplay.jsonl.gz"}
                and s.parent.name <= folder.name
                and not (s.suffix == ".gz" and s.with_suffix("").exists())
            ]
            selection = (make_replay_sqlite if args.initial_champion else make_replay)(
                sources,
                replay,
                args.window_games,
                min(int(args.replay_max_gib * 1024**3), budget.available_bytes()),
                check=budget.check,
                excluded_fens={row["fen"] for row in position_suite["evaluation"]},
                minimum_games=args.games if args.initial_champion else 1250,
            )
            save(folder / "replay-selection.json", selection)
            split_path = folder / "split.json"
            if args.initial_champion:
                from campaign_state import role_split

                role_split(
                    DatasetIndex(replay),
                    split_path,
                    args.training_games,
                    args.validation_games,
                    42,
                )
                if selection["games"] != args.games:
                    raise ValueError(
                        "Seeded collection must preserve exact requested game count"
                    )
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
                "--batch",
                args.batch,
                "--training-games",
                args.training_games,
                "--validation-games",
                args.validation_games,
                "--device",
                args.device,
            ]
            if checkpoint:
                cmd += ["--resume", checkpoint.with_suffix(".pt")]
            if args.initial_champion:
                cmd += [
                    "--split",
                    split_path,
                    "--epoch-state",
                    folder / "epoch-state.pt",
                ]
            run(*cmd, budget=budget)
        verify_manifest(json.loads((folder / "league.json").read_text()))
        accepted = True
        candidate_metadata = validate_checkpoint(
            model, args.training_games, args.validation_games
        )
        if args.initial_champion and not candidate_metadata.get("training_complete"):
            raise ValueError("Candidate training is incomplete")
        summary = None
        if args.initial_champion:
            parity_dir = folder / "parity-models"
            parity_dir.mkdir(exist_ok=True)
            for kind in ["nnue", "cnn"]:
                source = (
                    model if kind == args.kind else ROOT / "models" / f"{kind}.json"
                )
                for suffix in [".json", ".pt", ".meta.json"]:
                    shutil.copy2(
                        source.with_suffix(suffix), parity_dir / f"{kind}{suffix}"
                    )
            run(
                "python/verify_models.py",
                "--models",
                parity_dir,
                "--positions",
                80,
                budget=budget,
            )
        if os.environ.get("SKALLAGRIM_CAMPAIGN_BARRIER") == "1":
            identity = pin(model.with_suffix(".meta.json"))["sha256"]
            save(folder / "evaluation-ready.json", dict(identity=identity))
            while (
                not (folder / "evaluation-ack.json").exists()
                or json.loads((folder / "evaluation-ack.json").read_text()).get(
                    "identity"
                )
                != identity
            ):
                budget.check()
                time.sleep(1)
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
