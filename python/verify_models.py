"""Check Python/Rust encodings, values and every policy logit across a legal game."""

import argparse
import random
import json
from pathlib import Path
import torch
from models import CNN, NNUE, encode, policy_index
from transport import Engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--positions", type=int, default=24)
    args = p.parse_args()
    torch.set_num_threads(1)
    nn = json.loads((args.models / "nnue.json").read_text())
    cn = json.loads((args.models / "cnn.json").read_text())
    nnue = NNUE(nn["hidden"])
    cnn = CNN(cn["channels"], len(cn["blocks"]))
    nnue.load_state_dict(torch.load(args.models / "nnue.pt", weights_only=True))
    cnn.load_state_dict(torch.load(args.models / "cnn.pt", weights_only=True))
    nnue.eval()
    cnn.eval()
    rng = random.Random(73)
    largest = 0
    with Engine(
        options={
            "NNUEFile": str((args.models / "nnue.json").resolve()),
            "CNNFile": str((args.models / "cnn.json").resolve()),
        }
    ) as engine:
        for i in range(args.positions):
            state = engine.snapshot()
            if state["outcome"]:
                engine.position([])
                state = engine.snapshot()
            result = engine.evaluate()
            x, f = encode(state["fen"], state["history"])
            assert torch.equal(torch.tensor(result["planes"]), x.flatten())
            assert set(result["features"]) == set(f.nonzero().flatten().tolist())
            with torch.no_grad():
                n = nnue(f[None]).item()
                v, policy = cnn(x[None])
            errors = [
                abs(n - result["nnue"]),
                abs(v.item() - result["cnn"][0]),
                (policy.flatten() - torch.tensor(result["cnn"][1])).abs().max().item(),
            ]
            largest = max(largest, *errors)
            assert max(errors) < 2e-5, (i, errors)
            for move in state["moves"]:
                assert 0 <= policy_index(move) < 4840
            engine.play(rng.choice(state["moves"]))
    print(
        f"Verified {args.positions} positions: encodings, NNUE, CNN value and 4,840 logits; max error {largest:.3g}"
    )


if __name__ == "__main__":
    main()
