"""Train from outcome-labelled self-play JSONL; export both inference formats."""

import argparse
import json
from pathlib import Path
import random
import copy
import hashlib
import torch
from models import CNN, NNUE, encode, policy_index
from records import save
from dataset import DatasetIndex, open_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--kind", choices=["nnue", "cnn"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Permit fewer than 1,000 training games; output is never a strength candidate",
    )
    args = p.parse_args()
    if (
        args.epochs <= 0
        or args.batch <= 0
        or not 1 <= args.hidden <= 1024
        or not 1 <= args.channels <= 128
        or not 0 <= args.blocks <= 32
    ):
        raise SystemExit("invalid epochs, batch, or model dimensions")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    index = DatasetIndex(args.data)
    if not index.groups:
        raise SystemExit(
            "No terminal-labelled training samples. Generate completed self-play games first."
        )
    # Split by game, not position, to avoid leakage between correlated positions.
    games = sorted(index.groups)
    random.shuffle(games)
    if len(games) < 2:
        raise SystemExit(
            "Need at least two completed games for a game-level validation split."
        )
    valid = set(games[: max(1, len(games) // 5)])
    training = sorted(
        offset
        for game, offsets in index.groups.items()
        if game not in valid
        for offset in offsets
    )
    validation = sorted(
        offset
        for game, offsets in index.groups.items()
        if game in valid
        for offset in offsets
    )
    if len(games) - len(valid) < 1000 and not args.smoke:
        raise SystemExit(
            f"Need at least 1,000 completed TRAINING games plus held-out validation; found {len(games) - len(valid)} training games. --smoke is only for pipeline tests."
        )
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.5)
    net = NNUE(args.hidden) if args.kind == "nnue" else CNN(args.channels, args.blocks)
    if args.resume:
        net.load_state_dict(
            torch.load(args.resume, map_location="cpu", weights_only=True)
        )
    net.to(args.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    data_stream = open_data(args.data)

    def batch_loss(offsets):
        batch = DatasetIndex.read_batch(data_stream, offsets)
        enc = [encode(r["fen"], r["history"]) for r in batch]
        x = torch.stack([e[1 if args.kind == "nnue" else 0] for e in enc]).to(
            args.device
        )
        target = torch.tensor([r["target"] for r in batch], device=args.device)
        if args.kind == "nnue":
            return (net(x) - target).square().mean()
        value, policy = net(x)
        target_policy = torch.zeros_like(policy)
        legal = torch.zeros_like(policy, dtype=torch.bool)
        for i, r in enumerate(batch):
            for move in r["legal"]:
                legal[i, policy_index(move)] = True
            for move, weight in r["policy"]:
                target_policy[i, policy_index(move)] = weight
        # Legal-move normalization matches search and avoids learning impossible moves.
        logp = policy.masked_fill(~legal, -1e9).log_softmax(dim=1)
        return (value - target).square().mean() - (target_policy * logp).sum(
            dim=1
        ).mean()

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(args.epochs):
        random.shuffle(training)
        net.train()
        total = 0
        for start in range(0, len(training), args.batch):
            batch = training[start : start + args.batch]
            optimizer.zero_grad()
            loss = batch_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            total += loss.item() * len(batch)
        net.eval()
        with torch.no_grad():
            val = sum(
                batch_loss(validation[i : i + args.batch]).item()
                * len(validation[i : i + args.batch])
                for i in range(0, len(validation), args.batch)
            ) / len(validation)
        print(
            json.dumps(
                dict(
                    epoch=epoch + 1,
                    train_loss=total / len(training),
                    validation_loss=val,
                )
            ),
            flush=True,
        )
        if val < best_loss:
            best_loss = val
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch + 1
    if best_state is None:
        raise SystemExit("No finite validated checkpoint")
    net.load_state_dict(best_state)
    net.cpu().eval()
    net.export(args.output)
    checkpoint = args.output.with_suffix(".pt")
    staging = checkpoint.with_suffix(".pt.tmp")
    torch.save(net.state_dict(), staging)
    staging.replace(checkpoint)
    save(
        args.output.with_suffix(".meta.json"),
        dict(
            kind=args.kind,
            seed=args.seed,
            epochs=args.epochs,
            best_epoch=best_epoch,
            smoke=args.smoke,
            data_sha256=index.sha256,
            training_games=len(games) - len(valid),
            validation_games=len(valid),
            training_samples=len(training),
            validation_samples=len(validation),
            side_counts={
                side: sum(
                    counts[side]
                    for game, counts in index.sides.items()
                    if game not in valid
                )
                for side in ["a", "d"]
            },
            validation_loss=best_loss,
            model_artifacts={
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [args.output, checkpoint]
            },
            config=vars(args)
            | {
                "data": str(args.data),
                "output": str(args.output),
                "resume": str(args.resume),
            },
        ),
    )
    data_stream.close()


if __name__ == "__main__":
    main()
