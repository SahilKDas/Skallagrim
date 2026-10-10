"""Timing regression probe, not a playing-strength benchmark."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from records import save
from transport import Engine, ENGINE, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--cpu", type=int, default=6)
    p.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/deadline-regression.json"
    )
    args = p.parse_args()
    if args.trials <= 0:
        raise SystemExit("trials must be positive")
    fixture = ROOT / "runs/final-gauntlets/rho/beta-04.json"
    game = json.loads(fixture.read_text())
    history = game["opening"]["moves"] + [row["move"] for row in game["moves"]]
    results = []
    for bot in ["Rho", "Iota", "Beta"]:
        option, model = (
            ("CNNFile", "cnn.json") if bot == "Rho" else ("NNUEFile", "nnue.json")
        )
        with Engine(
            options={"Bot": bot, option: str(ROOT / "models" / model)}, cpu=args.cpu
        ) as engine:
            for label, moves in [("start", []), ("315-ply history", history)]:
                engine.position(moves)
                state = engine.snapshot()
                if state["outcome"]:
                    raise RuntimeError(
                        f"Timing fixture now terminal: {state['outcome']}"
                    )
                elapsed = []
                for _ in range(args.trials):
                    move, milliseconds, _, _ = engine.go(300, 2000)
                    assert move in state["moves"]
                    elapsed.append(milliseconds)
                row = dict(
                    bot=bot,
                    position=label,
                    trials=args.trials,
                    mean_ms=statistics.mean(elapsed),
                    max_ms=max(elapsed),
                    over_300=sum(v > 300 for v in elapsed),
                    over_325=sum(v > 325 for v in elapsed),
                    elapsed_ms=elapsed,
                )
                results.append(row)
                print(json.dumps(row), flush=True)
    save(
        args.output,
        dict(
            engine_sha256=hashlib.sha256(ENGINE.read_bytes()).hexdigest(),
            fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
            cpu=args.cpu,
            movetime_ms=300,
            diagnostic_allowance_ms=2000,
            strength_evidence=False,
            results=results,
        ),
    )


if __name__ == "__main__":
    main()
