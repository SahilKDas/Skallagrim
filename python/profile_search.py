"""Measure pinned engines at fixed clocks; diagnostic results are not promotion."""

import argparse
import json
from pathlib import Path
import time

from league import load_positions
from pin_campaign_baseline import pin
from records import save
from referee import Referee
from transport import Engine, pin_current_process


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--positions", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--engine", type=Path, help="Experimental binary; baseline weights stay fixed"
    )
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()
    pin_current_process()
    if args.trials < 1:
        raise ValueError("trials must be positive")
    baseline = json.loads(args.baseline.read_text())
    if pin(Path(baseline["engine"]["path"])) != baseline["engine"]:
        raise ValueError("Pinned engine changed")
    for champion in baseline["champions"].values():
        for artifact in champion["artifacts"].values():
            if pin(Path(artifact["path"])) != artifact:
                raise ValueError("Pinned model changed")
    suite = load_positions(args.positions)["evaluation"]
    rows = []
    for bot in ["iota", "beta", "rho"]:
        entry = baseline["config"]["engines"][bot]
        with (
            Engine(
                command=[str(args.engine.resolve())]
                if args.engine
                else entry["command"],
                options=entry["options"],
                cpu=6,
            ) as engine,
            Referee() as referee,
        ):
            for i, position in enumerate(suite):
                # Both side-to-move states are represented through a legal ply,
                # preserving history rather than flipping FEN side arbitrarily.
                engine.position(position["moves"], fen=position["start"])
                state = engine.snapshot()
                for side in range(2):
                    if state["outcome"]:
                        rows.append(dict(bot=bot, position=i, terminal=True))
                        break
                    referee.position(fen=state["fen"])
                    before = time.perf_counter()
                    evaluation = engine.evaluate()
                    eval_ms = (time.perf_counter() - before) * 1000
                    for trial in range(args.trials):
                        started = time.perf_counter()
                        try:
                            move, elapsed, _, lines = engine.go(300, 25)
                            if move not in state["moves"]:
                                raise ValueError("illegal search move")
                            row = dict(
                                bot=bot,
                                position=i,
                                side=state["side"],
                                trial=trial,
                                **engine.last_search_timing,
                            )
                            row["completed_alpha_beta_depth"] = (
                                row["depth"]
                                if bot == "iota" and not row["fallback"]
                                else None
                            )
                            row["mcts_simulations"] = (
                                row["nodes"]
                                if bot != "iota" and not row["fallback"]
                                else None
                            )
                        except TimeoutError:
                            # Drain only for cleanup, preserving the failed 325-ms measurement.
                            engine.until("bestmove", 5)
                            row = dict(
                                bot=bot,
                                position=i,
                                side=state["side"],
                                trial=trial,
                                deadline_missed=True,
                                transport_ms=(time.perf_counter() - started) * 1000,
                            )
                        row.update(
                            eval_roundtrip_ms=eval_ms,
                            inference=evaluation.get("timing"),
                            referee_latency=referee.latencies[-1],
                        )
                        rows.append(row)
                    state = engine.play(sorted(state["moves"])[0])
    save(
        args.output,
        dict(
            baseline=pin(args.baseline),
            positions=pin(args.positions),
            experimental_engine=pin(args.engine) if args.engine else None,
            movetime_ms=300,
            ipc_allowance_ms=25,
            cpu=6,
            results=rows,
            strength_evidence=False,
            limitations=[
                "Inference diagnostic rebuild timing is not cumulative search inference time",
                "Referee timing is measured separately from engine transport; no clocks relaxed",
            ],
        ),
    )


if __name__ == "__main__":
    main()
