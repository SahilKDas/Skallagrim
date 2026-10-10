"""Generate and validate deterministic short legal opening sequences."""

import argparse
import random
from transport import Engine, ROOT
from models import save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20261009)
    args = p.parse_args()
    rng = random.Random(args.seed)
    openings = [dict(id="standard", moves=[])]
    with Engine() as engine:
        for i in range(1, 25):
            while True:
                engine.position([])
                state = engine.snapshot()
                moves = []
                for _ in range(2 + (i % 4) * 2):
                    move = rng.choice(state["moves"])
                    moves.append(move)
                    state = engine.play(move)
                    if state["outcome"]:
                        break
                if not state["outcome"] and not any(
                    o["moves"] == moves for o in openings
                ):
                    break
            openings.append(dict(id=f"seeded-{i:02d}", moves=moves))
    save(
        ROOT / "benchmarks/openings.json",
        dict(
            format="skallagrim-openings-v1",
            seed=args.seed,
            description="25 deterministic legal starts; role-swapped pairs; not a curated balanced opening book",
            openings=openings,
        ),
    )


if __name__ == "__main__":
    main()
