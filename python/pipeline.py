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
from pathlib import Path
from records import save
from transport import ROOT
from dataset import make_replay, archive_data
from resources import StorageBudget


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
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["cnn", "nnue"], required=True)
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
    if args.games < 1250 or args.window_games < 1250:
        raise SystemExit("Use >=1,250 games: at least 1,000 training plus validation.")
    if args.replay_max_gib <= 0:
        raise SystemExit("--replay-max-gib must be positive")
    if args.max_storage_gib <= 0:
        raise SystemExit("--max-storage-gib must be positive")
    args.run = args.run.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    budget = StorageBudget(args.run, int(args.max_storage_gib * 1024**3))
    save(args.run / "config.json", vars(args) | {"run": str(args.run)})
    checkpoint = None
    for generation in range(args.generations):
        folder = args.run / f"generation-{generation:04d}"
        folder.mkdir(exist_ok=True)
        samples = folder / "selfplay.jsonl"
        model = folder / f"{args.kind}.json"
        # Validated completed generations can be resumed without retraining.
        if (folder / "complete.json").exists():
            checkpoint = Path(
                json.loads((folder / "complete.json").read_text())["promoted_model"]
            )
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
        accepted = True
        summary = None
        if checkpoint is not None:
            config = json.loads((ROOT / "benchmarks/engines.json").read_text())
            template = config["engines"]["rho" if args.kind == "cnn" else "beta"]
            engines = {}
            for name, path in [("candidate", model), ("champion", checkpoint)]:
                entry = json.loads(json.dumps(template))
                option = "CNNFile" if args.kind == "cnn" else "NNUEFile"
                entry["options"][option] = str(path.resolve())
                meta = str(path.with_suffix(".meta.json").resolve())
                entry["artifacts"] = [
                    "target/release/skallagrim.exe",
                    str(path.resolve()),
                    meta,
                ]
                entry["training_metadata"] = [meta]
                engines[name] = entry
            gate_config = folder / "gate-config.json"
            save(gate_config, dict(engines=engines))
            run(
                "python/benchmark.py",
                "--config",
                gate_config,
                "--candidate",
                "candidate",
                "--opponents",
                "champion",
                "--output",
                folder / "gate",
                "--resume",
                budget=budget,
            )
            summary = json.loads((folder / "gate/champion-summary.json").read_text())
            failures = sum(
                summary["termination_counts"].get(key, 0)
                for key in [
                    "timeout",
                    "engine-failure",
                    "illegal-move",
                    "rules-incompatibility",
                ]
            )
            accepted = (
                summary["paired_bootstrap_95"][0] > 0.5
                and all(v["score_rate"] >= 0.5 for v in summary["by_side"].values())
                and failures == 0
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
                promotion="50-game/300-ms internal gate"
                if checkpoint
                else "initial bootstrap; external strength unproven",
            ),
        )
        checkpoint = promoted
        if not args.retain_raw:
            archive_data(samples, budget.check)
            archive_data(folder / "replay.jsonl", budget.check)


if __name__ == "__main__":
    main()
