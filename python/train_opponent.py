"""Reproducible supervised warm starts for upstream engines without public weights.

These are explicitly OUR trained checkpoints, not published author-strength nets.
Uses upstream encoders/networks and >=1,000 completed training games.
"""

import argparse
import contextlib
import copy
import hashlib
import json
import random
import sys
from collections import Counter, deque
from pathlib import Path
import torch
from models import save
from transport import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["luizz", "nicholas"], required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.5)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    rng = random.Random(args.seed)
    sys.path.insert(0, str(ROOT / "external" / args.kind))
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line]
    rows = [r for r in rows if r.get("target") is not None]
    games = sorted({r["game"] for r in rows})
    rng.shuffle(games)
    held = set(games[: max(1, len(games) // 5)])
    if len(games) - len(held) < 1000:
        raise SystemExit("Need >=1,000 completed training games plus validation")
    train = [r for r in rows if r["game"] not in held]
    valid = [r for r in rows if r["game"] in held]

    def grid(fen):
        result = []
        for rank in fen.split()[0].split("/"):
            row = []
            digits = ""
            for c in rank + "!":
                if c.isdigit():
                    digits += c
                    continue
                if digits:
                    row.extend([0] * int(digits))
                    digits = ""
                if c != "!":
                    row.append(".adk".index(c))
            result.append(row)
        return __import__("numpy").array(result, dtype="int8")

    with contextlib.redirect_stdout(sys.stderr):
        if args.kind == "luizz":
            from hnefatafl.game import HnefataflGame, Player, Move, encode_move
            from hnefatafl.network import HnefataflNetwork

            net = HnefataflNetwork(args.channels, args.blocks)

            def encode(row):
                state = HnefataflGame()
                state.board = grid(row["fen"])
                state.current_player = (
                    Player.ATTACKER if row["fen"].endswith(" a") else Player.DEFENDER
                )
                # Upstream Lua encoder intentionally leaves all history planes zero.
                from adapters.neural_opponent import split_move

                return torch.tensor(state.encode_state()), lambda m: encode_move(
                    Move(*split_move(m))
                )
        else:
            from hnefatafl.core.gamestate_v2 import GameState
            from hnefatafl.core.board import Board
            from hnefatafl.core.gameTypes import Player, Point
            from hnefatafl.core.move import Move
            from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
            from hnefatafl.zero.lightning_network import DualNetwork
            from adapters.neural_opponent import split_move

            encoder = SevenPlaneEncoder(11)
            net = DualNetwork(encoder)

            def encode(row):
                board = Board(11)
                board.grid = grid(row["fen"])
                state = GameState(
                    board,
                    Player.black if row["fen"].endswith(" a") else Player.white,
                    None,
                    None,
                    history=Counter(),
                    move_count=row["ply"],
                    max_moves=10000,
                    board_history=deque(
                        (
                            [
                                __import__("numpy").zeros((11, 11), dtype="int8")
                                for _ in range(max(0, 8 - len(row["history"][:-1])))
                            ]
                            + [grid(f) for f in row["history"][:-1]][-8:]
                        ),
                        maxlen=8,
                    ),
                )

                def index(move):
                    a, b, c, d = split_move(move)
                    return encoder.encode_move(Move(Point(a, b), Point(c, d)))

                return torch.tensor(encoder.encode(state), dtype=torch.float32), index

        net.to(args.device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        # Materialize only encoded tensors; avoid repeated Python encoding each epoch.
        encoded = {}
        for row in rows:
            x, index = encode(row)
            target = [(index(move), weight) for move, weight in row["policy"]]
            legal = [index(move) for move in row["legal"]]
            value = row["target"] * (1 if row["fen"].endswith(" a") else -1)
            encoded[id(row)] = (x.to(torch.uint8), target, value, legal)
            row.pop("history", None)
            row.pop("legal", None)
            row.pop("policy", None)

    def loss(batch):
        values = [encoded[id(r)] for r in batch]
        x = torch.stack([v[0] for v in values]).to(
            device=args.device, dtype=torch.float32
        )
        target = torch.zeros((len(batch), 4840), device=args.device)
        legal = torch.zeros_like(target, dtype=torch.bool)
        for i, v in enumerate(values):
            legal[i, v[3]] = True
            for index, weight in v[1]:
                target[i, index] = weight
        value_target = torch.tensor([v[2] for v in values], device=args.device)
        policy, value = net(x)
        return (
            -(target * policy.masked_fill(~legal, -1e9).log_softmax(dim=1))
            .sum(dim=1)
            .mean()
            + (value.flatten() - value_target).square().mean()
        )

    best_loss = float("inf")
    best = None
    best_epoch = 0
    for epoch in range(args.epochs):
        rng.shuffle(train)
        net.train()
        for i in range(0, len(train), args.batch):
            optimizer.zero_grad()
            l = loss(train[i : i + args.batch])
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
        net.eval()
        with torch.no_grad():
            score = sum(
                loss(valid[i : i + args.batch]).item() * len(valid[i : i + args.batch])
                for i in range(0, len(valid), args.batch)
            ) / len(valid)
        print(
            json.dumps(dict(kind=args.kind, epoch=epoch + 1, validation_loss=score)),
            flush=True,
        )
        if score < best_loss:
            best_loss = score
            best = copy.deepcopy(net.state_dict())
            best_epoch = epoch + 1
    if best is None:
        raise SystemExit("No finite validation checkpoint")
    net.load_state_dict(best)
    net.cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        dict(
            model_state_dict=net.state_dict(),
            num_channels=args.channels,
            num_res_blocks=args.blocks,
            board_size=11,
            epoch=best_epoch,
        )
        if args.kind == "luizz"
        else dict(state_dict=net.state_dict())
    )
    torch.save(checkpoint, args.output)
    save(
        args.output.with_suffix(".meta.json"),
        dict(
            provenance="Skallagrim supervised bootstrap, not upstream-published weights",
            kind=args.kind,
            device=args.device,
            training_games=len(games) - len(held),
            validation_games=len(held),
            best_epoch=best_epoch,
            validation_loss=best_loss,
            channels=args.channels if args.kind == "luizz" else 64,
            blocks=args.blocks if args.kind == "luizz" else 4,
            seed=args.seed,
            data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
            side_counts={
                side: sum(r["fen"].endswith(" " + side) for r in train)
                for side in ["a", "d"]
            },
            history="upstream encoding: Lua zero history planes; Nichols zero-padded eight prior boards",
            rules_audited=False,
        ),
    )


if __name__ == "__main__":
    main()
