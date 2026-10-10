"""Mine reproducible tactical starts from new legal random continuations.

Positions originate at standard setup, not arbitrary synthetic boards. Each
source game belongs to one partition; new evaluation roots are held out from
subsequent league training, including accidental visits during self-play.
"""

import argparse
import random
from pathlib import Path
from transport import Engine, ROOT
from referee import Referee
from league import RULES, position_id
from records import save


def tactical_score(fen):
    board = []
    for rank in fen.split()[0].split("/"):
        digits = ""
        for ch in rank + "!":
            if ch.isdigit():
                digits += ch
            else:
                if digits:
                    board.extend("." * int(digits))
                    digits = ""
                if ch != "!":
                    board.append(ch)
    king = board.index("k")
    x, y = king % 11, king // 11
    nearby = sum(
        ch == "a" and abs(i % 11 - x) + abs(i // 11 - y) <= 3
        for i, ch in enumerate(board)
    )
    edge = min(x, y, 10 - x, 10 - y)
    captures = 36 - board.count("a") - board.count("d")
    return 3 * nearby + captures + 2 * (5 - edge)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/league-positions.json"
    )
    p.add_argument("--seed", type=int, default=202610091)
    p.add_argument("--positions", type=int, default=25)
    args = p.parse_args()
    if args.positions < 25:
        raise SystemExit("Need >=25 positions per partition")
    rng = random.Random(args.seed)
    result = dict(
        format="skallagrim-league-positions-v1",
        seed=args.seed,
        training=[],
        evaluation=[],
        description="Heuristically difficult legal continuations; difficulty is not a solved tactical label",
    )
    seen = set()
    with (
        Engine(options={"Bot": "Baseline"}) as engine,
        Referee(log_dir=ROOT / "runs/league-position-validation") as ref,
    ):
        for source in range(args.positions * 20):
            partition = "training" if source % 2 == 0 else "evaluation"
            if len(result[partition]) >= args.positions:
                continue
            engine.position([])
            start = engine.snapshot()["fen"]
            moves, candidates = [], []
            for ply in range(180):
                state = engine.snapshot()
                if state["outcome"]:
                    break
                if ply >= 20 and ply % 2 == 0 and state["fen"] not in seen:
                    candidates.append(
                        (tactical_score(state["fen"]), state["fen"], list(moves))
                    )
                move = rng.choice(state["moves"])
                engine.play(move)
                moves.append(move)
            if not candidates:
                continue
            score, fen, history = max(candidates)
            verified = ref.position(history, start)
            if (
                verified["outcome"]
                or verified["fen"] != fen
                or verified["rules"] != RULES
            ):
                raise RuntimeError("Position mining disagrees with referee")
            row = dict(
                rules=RULES,
                start=start,
                moves=history,
                fen=fen,
                provenance=dict(
                    method="seeded legal random continuation, highest tactical pressure",
                    seed=args.seed,
                    source_game=source,
                    pressure_score=score,
                ),
            )
            row["id"] = position_id(row)
            result[partition].append(row)
            seen.add(fen)
            if all(
                len(result[k]) == args.positions for k in ["training", "evaluation"]
            ):
                break
    if any(len(result[k]) != args.positions for k in ["training", "evaluation"]):
        raise RuntimeError("Could not assemble enough disjoint positions")
    save(args.output, result)
    print(
        f"Saved {args.positions} training and {args.positions} held-out positions to {args.output}"
    )


if __name__ == "__main__":
    main()
