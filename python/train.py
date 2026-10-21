"""Train from outcome-labelled self-play JSONL; export both inference formats."""

import argparse
import json
from pathlib import Path
import random
import copy
import hashlib
import os
import math
import time
import torch
from models import CNN, NNUE, encode, encode_nnue, migrate_nnue, policy_index
from nnue_experiments import augment_row, target_for, training_offsets, validate_labels
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
    p.add_argument("--split", type=Path)
    p.add_argument("--training-games", type=int, default=1000)
    p.add_argument("--validation-games", type=int, default=1)
    p.add_argument("--epoch-state", type=Path)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--min-improvement", type=float, default=0)
    p.add_argument("--feature-version", choices=["v1", "v2"], default="v1")
    p.add_argument("--migrate-nnue", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--samples-per-game", type=int, default=0)
    p.add_argument("--search-labels", type=Path)
    p.add_argument("--target-mix", type=float, default=0)
    p.add_argument("--deadline-unix", type=float)
    args = p.parse_args()
    if args.deadline_unix is not None and not math.isfinite(args.deadline_unix):
        raise ValueError("Training deadline must be finite")
    data_stream = None

    def check_budget():
        if args.deadline_unix is not None and time.time() >= args.deadline_unix:
            if data_stream is not None:
                data_stream.close()
            save(
                args.output.with_suffix(".budget.json"),
                dict(
                    complete=False,
                    reason="training-time-budget",
                    partial_epoch_discarded=True,
                ),
            )
            raise SystemExit(75)

    check_budget()
    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
        or args.patience < 0
        or not math.isfinite(args.min_improvement)
        or args.min_improvement < 0
        or args.samples_per_game < 0
        or not 0 <= args.target_mix <= 1
    ):
        raise ValueError("Invalid experimental training controls")
    if args.kind != "nnue" and (
        args.augment
        or args.feature_version != "v1"
        or args.migrate_nnue
        or args.target_mix
    ):
        raise ValueError("NNUE experiment options cannot be applied to CNN")
    if args.target_mix and not args.search_labels:
        raise ValueError("Mixed targets require a search-label manifest")
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
    if args.split:
        split = json.loads(args.split.read_text())
        if (
            split["data_sha256"] != index.sha256
            or set(split["training"]) & set(split["validation"])
            or set(split["training"]) | set(split["validation"]) != set(games)
        ):
            raise ValueError("Changed, overlapping or incomplete game split")
        valid = set(split["validation"])
    else:
        valid = set(games[: max(1, len(games) // 5)])
    training = training_offsets(index, valid, args.samples_per_game, random)
    validation = sorted(
        offset
        for game, offsets in index.groups.items()
        if game in valid
        for offset in offsets
    )
    if (
        len(games) - len(valid) < max(1000, args.training_games)
        or len(valid) < args.validation_games
    ) and not args.smoke:
        raise SystemExit(
            f"Need at least {max(1000, args.training_games):,} completed TRAINING games plus {args.validation_games} validation; found {len(games) - len(valid)} training games. --smoke is only for pipeline tests."
        )
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.5)
    check_budget()
    labels = (
        validate_labels(
            json.loads(args.search_labels.read_text()), index.sha256, set(games) - valid
        )
        if args.search_labels
        else {}
    )
    net = (
        NNUE(args.hidden, args.feature_version)
        if args.kind == "nnue"
        else CNN(args.channels, args.blocks)
    )
    if args.resume:
        initial = torch.load(args.resume, map_location="cpu", weights_only=True)
        if args.migrate_nnue:
            migrate_nnue(net, initial)
        else:
            net.load_state_dict(initial)
    net.to(args.device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    data_stream = open_data(args.data)

    def batch_loss(offsets, training_batch=False):
        check_budget()
        batch = DatasetIndex.read_batch(data_stream, offsets)
        targets = [
            target_for(r, labels, args.target_mix) if training_batch else r["target"]
            for r in batch
        ]
        if training_batch and args.augment:
            batch = [augment_row(r, random.randrange(8)) for r in batch]
        x = torch.stack(
            [encode_nnue(r["fen"], r["history"], args.feature_version) for r in batch]
            if args.kind == "nnue"
            else [encode(r["fen"], r["history"])[0] for r in batch]
        ).to(args.device)
        target = torch.tensor(targets, device=args.device)
        if args.kind == "nnue":
            return (net(x) - target).square().mean()
        value, policy = net(x)
        # Build sparse labels on CPU, then transfer once. Thousands of scalar
        # CUDA assignments per batch otherwise dominate this tiny network.
        target_policy = torch.zeros(policy.shape, dtype=policy.dtype, device="cpu")
        legal = torch.zeros(policy.shape, dtype=torch.bool, device="cpu")
        for i, r in enumerate(batch):
            for move in r["legal"]:
                legal[i, policy_index(move)] = True
            for move, weight in r["policy"]:
                target_policy[i, policy_index(move)] = weight
        # Legal-move normalization matches search and avoids learning impossible moves.
        legal = legal.to(args.device)
        target_policy = target_policy.to(args.device)
        logp = policy.masked_fill(~legal, -1e9).log_softmax(dim=1)
        return (value - target).square().mean() - (target_policy * logp).sum(
            dim=1
        ).mean()

    def atomic_torch_save(value, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(path.suffix + ".tmp")
        with staging.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        staging.replace(path)

    def atomic_export(model):
        staging = args.output.with_suffix(".json.tmp")
        model.export(staging)
        with staging.open("rb+") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        staging.replace(args.output)

    identity = dict(
        kind=args.kind,
        data=index.sha256,
        data_format=args.data.suffix,
        split=hashlib.sha256(args.split.read_bytes()).hexdigest()
        if args.split
        else None,
        seed=args.seed,
        epochs=args.epochs,
        batch=args.batch,
        hidden=args.hidden,
        channels=args.channels,
        blocks=args.blocks,
        device=args.device,
        training_games=args.training_games,
        validation_games=args.validation_games,
        initial_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest()
        if args.resume
        else None,
    )
    # Preserve historical default identities for compatible old epoch resumes.
    if (
        args.learning_rate != 1e-3
        or args.weight_decay != 0.01
        or args.patience
        or args.min_improvement
        or args.feature_version != "v1"
        or args.migrate_nnue
        or args.augment
        or args.samples_per_game
        or args.search_labels
        or args.target_mix
    ):
        identity["experiment"] = dict(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_improvement=args.min_improvement,
            feature_version=args.feature_version,
            migrate=args.migrate_nnue,
            augment=args.augment,
            samples_per_game=args.samples_per_game,
            target_mix=args.target_mix,
            labels=hashlib.sha256(args.search_labels.read_bytes()).hexdigest()
            if args.search_labels
            else None,
        )
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    begin = 0
    metrics = []
    stale, significant_loss = 0, float("inf")
    if args.epoch_state and args.epoch_state.exists():
        state = torch.load(
            args.epoch_state, map_location=args.device, weights_only=True
        )
        if state["identity"] != identity:
            raise ValueError(
                "Epoch resume rejected: changed data, checkpoint, split or settings"
            )
        net.load_state_dict(state["net"])
        optimizer.load_state_dict(state["optimizer"])
        begin, best_loss, best_epoch = (
            state["epoch"],
            state["best_loss"],
            state["best_epoch"],
        )
        best_state, metrics, training = (
            state["best_state"],
            state["metrics"],
            state["training_order"],
        )
        stale = state.get("stale", 0)
        significant_loss = state.get("significant_loss", best_loss)
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all([row.cpu() for row in state["cuda_rng"]])
    if best_state is not None:
        # Restore the best export too: an interrupted later epoch may have
        # published a newer preview before its epoch transaction committed.
        preview = copy.deepcopy(net)
        preview.load_state_dict(best_state)
        preview.cpu().eval()
        atomic_export(preview)
        atomic_torch_save(preview.state_dict(), args.output.with_suffix(".pt"))
    for epoch in range(begin, args.epochs):
        if args.patience and stale >= args.patience:
            break
        check_budget()
        epoch_started = time.monotonic()
        random.shuffle(training)
        net.train()
        total = 0
        for start in range(0, len(training), args.batch):
            batch = training[start : start + args.batch]
            optimizer.zero_grad()
            loss = batch_loss(batch, training_batch=True)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Nonfinite training loss; previous durable epoch retained"
                )
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
        if not math.isfinite(val):
            raise RuntimeError(
                "Nonfinite validation loss; previous durable epoch retained"
            )
        metric = dict(
            epoch=epoch + 1,
            train_loss=total / len(training),
            validation_loss=val,
            seconds=time.monotonic() - epoch_started,
        )
        metrics.append(metric)
        print(json.dumps(metric), flush=True)
        if val < significant_loss - args.min_improvement:
            significant_loss, stale = val, 0
        else:
            stale += 1
        if val < best_loss:
            best_loss, best_epoch = val, epoch + 1
            best_state = copy.deepcopy(net.state_dict())
            preview = copy.deepcopy(net).cpu().eval()
            atomic_export(preview)
            atomic_torch_save(preview.state_dict(), args.output.with_suffix(".pt"))
        if args.epoch_state:
            atomic_torch_save(
                dict(
                    identity=identity,
                    epoch=epoch + 1,
                    net=net.state_dict(),
                    optimizer=optimizer.state_dict(),
                    best_loss=best_loss,
                    best_epoch=best_epoch,
                    best_state=best_state,
                    metrics=metrics,
                    stale=stale,
                    significant_loss=significant_loss,
                    training_order=training,
                    python_rng=random.getstate(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all()
                    if args.device.startswith("cuda")
                    else [],
                ),
                args.epoch_state,
            )
        save(
            args.output.with_suffix(".progress.json"),
            dict(
                epoch=epoch + 1,
                epochs=args.epochs,
                best_epoch=best_epoch,
                best_loss=best_loss,
                training_games=len(games) - len(valid),
                validation_games=len(valid),
                metrics=metrics,
                complete=False,
                identity=identity,
                model_artifacts={
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [args.output, args.output.with_suffix(".pt")]
                },
            ),
        )
    if best_state is None:
        raise SystemExit("No finite validated checkpoint")
    net.load_state_dict(best_state)
    net.cpu().eval()
    atomic_export(net)
    checkpoint = args.output.with_suffix(".pt")
    atomic_torch_save(net.state_dict(), checkpoint)
    save(
        args.output.with_suffix(".meta.json"),
        dict(
            kind=args.kind,
            seed=args.seed,
            epochs=args.epochs,
            completed_epochs=len(metrics),
            early_stopped=len(metrics) < args.epochs,
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
            training_complete=True,
            split_sha256=identity["split"],
            validation_roles={
                role: sum(
                    index.roles.get(game) == role for game in games if game in valid
                )
                for role in ("Attacker", "Defender")
            },
            candidate_roles={
                role: sum(
                    index.roles.get(game) == role for game in games if game not in valid
                )
                for role in ("Attacker", "Defender")
            },
            config=vars(args)
            | {
                "data": str(args.data),
                "output": str(args.output),
                "resume": str(args.resume),
                "split": str(args.split),
                "epoch_state": str(args.epoch_state),
                "search_labels": str(args.search_labels),
            },
        ),
    )
    save(
        args.output.with_suffix(".progress.json"),
        dict(
            epoch=len(metrics),
            epochs=args.epochs,
            best_epoch=best_epoch,
            best_loss=best_loss,
            metrics=metrics,
            complete=True,
            training_games=len(games) - len(valid),
            validation_games=len(valid),
        ),
    )
    data_stream.close()


if __name__ == "__main__":
    main()
