"""UCI bridge around upstream Python search, with a cooperative wall-clock loop.

The simulation iterator is timed; Luizz child states are materialized lazily.
Upstream selection, priors, evaluation, visit order and backups are retained.
No random network substitutes for missing weights. stdout is reserved for UCI.
"""

import argparse
import builtins
import contextlib
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import square

SIMULATIONS = 1_000_000_000
deadline = 0


def timed_range(*args):
    if args == (SIMULATIONS,):

        def iterations():
            for i in builtins.range(SIMULATIONS):
                if time.perf_counter() >= deadline:
                    break
                yield i

        return iterations()
    return builtins.range(*args)


def coord(row, col):
    return f"{chr(97 + col)}{row + 1}"


def split_move(move):
    i = next(i for i in range(1, len(move)) if move[i].isalpha())
    a, b = square(move[:i]), square(move[i:])
    return a // 11, a % 11, b // 11, b % 11


def main():
    global deadline
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["luizz", "nicholas"], required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument(
        "--eager",
        action="store_true",
        help="Keep native eager Luizz state copies for diagnostic timing",
    )
    p.add_argument("--reserve-ms", type=int, default=30)
    args = p.parse_args()
    if not args.weights.is_file():
        raise SystemExit(
            f"Pretrained weights required: {args.weights}. No substitute network will be used."
        )
    sys.path.insert(0, str(args.repo.resolve()))
    import torch

    torch.set_num_threads(1)
    # The upstream Luizz implementation auto-selects CUDA. Force CPU consistently.
    torch.cuda.is_available = lambda: False
    with contextlib.redirect_stdout(sys.stderr):
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
        if args.kind == "luizz":
            from hnefatafl.game import HnefataflGame, Move
            from hnefatafl.network import HnefataflNetwork
            from hnefatafl.mcts import MCTS

            net = HnefataflNetwork(
                num_channels=checkpoint["num_channels"],
                num_res_blocks=checkpoint["num_res_blocks"],
                board_size=checkpoint.get("board_size", 11),
            )
            net.load_state_dict(checkpoint["model_state_dict"])
            net.eval()
            module = importlib.import_module("hnefatafl.mcts")
            module.range = timed_range
            if not args.eager:
                from luizz_lazy import install

                install(module)
            agent = MCTS(neural_network=net, num_simulations=SIMULATIONS, temperature=0)

            def new():
                return HnefataflGame()

            def apply(state, move):
                m = Move(*split_move(move))
                if m not in state.get_legal_moves():
                    raise ValueError("upstream rules rejected history move " + move)
                state.make_move(m)
                return state

            def choose(state):
                move, _ = agent.search(state, temperature=0)
                return coord(move.from_row, move.from_col) + coord(
                    move.to_row, move.to_col
                )

            def legal(state):
                return [
                    coord(m.from_row, m.from_col) + coord(m.to_row, m.to_col)
                    for m in state.get_legal_moves()
                ]
        else:
            from hnefatafl.core.gamestate_v2 import GameState
            from hnefatafl.core.gameTypes import Point
            from hnefatafl.core.move import Move
            from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
            from hnefatafl.zero.lightning_network import DualNetwork
            from hnefatafl.zero.zeroagent_fast import ZeroAgent

            encoder = SevenPlaneEncoder(11)
            net = DualNetwork(encoder)
            net.load_state_dict(checkpoint.get("state_dict", checkpoint))
            net.eval()
            module = importlib.import_module("hnefatafl.zero.zeroagent_fast")
            module.range = timed_range
            agent = ZeroAgent(net, encoder, rounds_per_move=SIMULATIONS)

            def new():
                agent.clear_cache()
                return GameState.new_game(11, max_moves=10000)

            def apply(state, move):
                a, b, c, d = split_move(move)
                m = Move(Point(a, b), Point(c, d))
                if m not in state.get_legal_moves():
                    raise ValueError("upstream rules rejected history move " + move)
                return state.apply_move(m)

            def choose(state):
                # Avoid upstream history-blind inference cache contamination.
                agent.clear_cache()
                m = agent.select_move(state, temperature=0, add_noise=False)
                return (
                    "0000"
                    if m is None
                    else coord(m.from_pos.row, m.from_pos.col)
                    + coord(m.to_pos.row, m.to_pos.col)
                )

            def legal(state):
                return [
                    coord(m.from_pos.row, m.from_pos.col)
                    + coord(m.to_pos.row, m.to_pos.col)
                    for m in state.get_legal_moves()
                ]

        # Warm inference/JIT before the benchmark clock starts.
        state = new()
        # Warm actual transitions as well as inference; a clock-limited warmup
        # may expire before Numba has compiled the first apply_move call.
        deadline = float("inf")
        if args.kind == "nicholas":
            agent.num_rounds = 8
            choose(state)
            agent.num_rounds = SIMULATIONS
        else:
            agent.num_simulations = 8
            choose(state)
            agent.num_simulations = SIMULATIONS
        state = new()
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        try:
            with contextlib.redirect_stdout(sys.stderr):
                cmd = parts[0]
                output = None
                if cmd == "uci":
                    output = f"id name {args.kind} upstream neural adapter\nuciok"
                elif cmd == "isready":
                    output = "readyok"
                elif cmd == "ucinewgame":
                    state = new()
                elif cmd == "position":
                    if parts[1] != "startpos":
                        raise ValueError(
                            "adapter supports startpos + moves; arbitrary FEN is not audited"
                        )
                    state = new()
                    if "moves" in parts:
                        for move in parts[parts.index("moves") + 1 :]:
                            state = apply(state, move)
                elif cmd == "legal":
                    output = "legal " + " ".join(legal(state))
                elif cmd == "go":
                    ms = int(parts[parts.index("movetime") + 1])
                    deadline = time.perf_counter() + max(0, ms - args.reserve_ms) / 1000
                    output = "bestmove " + choose(state)
                elif cmd == "quit":
                    break
                elif cmd == "setoption":
                    raise ValueError("adapter has no runtime options")
                elif cmd == "stop":
                    pass
                else:
                    raise ValueError("unsupported command")
            if output:
                print(output, flush=True)
        except Exception as exc:
            print("info string error " + str(exc), flush=True)


if __name__ == "__main__":
    main()
