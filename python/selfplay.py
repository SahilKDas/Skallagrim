"""Generate authoritative, outcome-labelled samples; truncated games stay unlabelled."""

import argparse
import json
import random
from pathlib import Path
from transport import Engine


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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exploration", type=float, default=0.15)
    args = p.parse_args()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    options = {"Bot": args.bot}
    if args.nnue:
        options["NNUEFile"] = str(args.nnue.resolve())
    if args.cnn:
        options["CNNFile"] = str(args.cnn.resolve())
    with (
        Engine(options=options) as engine,
        args.output.open("w", encoding="utf8") as out,
    ):
        completed = 0
        game = 0
        while completed < args.games:
            engine.send("ucinewgame")
            engine.ready()
            samples = []
            state = engine.snapshot()
            for ply in range(args.max_plies):
                if state["outcome"]:
                    break
                best, elapsed, policy, _ = engine.go(args.movetime)
                if best not in state["moves"]:
                    raise RuntimeError("self-play returned illegal move")
                move = (
                    rng.choice(state["moves"])
                    if rng.random() < args.exploration
                    else best
                )
                samples.append(
                    dict(
                        format="skallagrim-sample-v1",
                        game=f"{args.seed}:{game}",
                        fen=state["fen"],
                        history=state["history"],
                        legal=state["moves"],
                        policy=policy or [[best, 1.0]],
                        played_move=move,
                        bot=args.bot,
                        ply=ply,
                    )
                )
                state = engine.play(move)
            outcome = state["outcome"]
            target = (
                None
                if outcome is None
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
                    adjudication="terminal" if outcome else "truncated",
                )
                out.write(json.dumps(row) + "\n")
            completed += outcome is not None
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
