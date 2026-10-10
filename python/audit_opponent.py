"""Focused differential rules audit of the pinned Python engines.

Agreement is evidence, not federation certification. Mismatches retain full FENs.
"""

import argparse
import contextlib
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from records import save
from transport import Engine, ROOT


def fen(pieces, side="a"):
    board = ["."] * 121
    board[60] = "k"
    for square, piece in pieces.items():
        board[square] = piece
    ranks = []
    for y in range(11):
        row = ""
        empty = 0
        for piece in board[y * 11 : y * 11 + 11] + ["!"]:
            if piece == ".":
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            if piece != "!":
                row += piece
        ranks.append(row)
    return "/".join(ranks) + " " + side


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["luizz", "nicholas"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(ROOT / "external" / args.kind))
    with contextlib.redirect_stdout(sys.stderr):
        from adapters.neural_opponent import split_move, coord

        if args.kind == "luizz":
            from hnefatafl.game import HnefataflGame, Move, Player

            def make(board, side):
                state = HnefataflGame()
                state.board = board
                state.current_player = (
                    Player.ATTACKER if side == "a" else Player.DEFENDER
                )
                return state

            def legal(state):
                return {
                    coord(m.from_row, m.from_col) + coord(m.to_row, m.to_col)
                    for m in state.get_legal_moves()
                }

            def apply(state, move):
                state.make_move(Move(*split_move(move)))
                return state

            def grid(state):
                return state.board

            def outcome(state):
                return (
                    None
                    if int(state.result) == 0
                    else "Defender"
                    if int(state.result) == 1
                    else "Attacker"
                    if int(state.result) == 2
                    else "Draw"
                )
        else:
            from hnefatafl.core.gamestate_v2 import GameState
            from hnefatafl.core.board import Board
            from hnefatafl.core.gameTypes import Player, Point
            from hnefatafl.core.move import Move

            def make(board, side):
                b = Board(11)
                b.grid = board
                return GameState(
                    b,
                    Player.black if side == "a" else Player.white,
                    None,
                    None,
                    history=Counter(),
                    max_moves=10000,
                )

            def legal(state):
                return {
                    coord(m.from_pos.row, m.from_pos.col)
                    + coord(m.to_pos.row, m.to_pos.col)
                    for m in state.get_legal_moves()
                }

            def apply(state, move):
                a, b, c, d = split_move(move)
                return state.apply_move(Move(Point(a, b), Point(c, d)))

            def grid(state):
                return state.board.grid

            def outcome(state):
                if not state.is_over():
                    return None
                return (
                    "Attacker"
                    if state.winner == Player.black
                    else "Defender"
                    if state.winner == Player.white
                    else "Draw"
                )

    from audit_dataset import board as unpack

    fixtures = [
        ("sandwich", fen({23: "a", 25: "d", 26: "a"}), "b3c3"),
        (
            "strict-shieldwall-support",
            fen({1: "d", 2: "d", 3: "a", 12: "a", 24: "a"}),
            "c3c2",
        ),
        ("shieldwall-flank", fen({1: "d", 2: "d", 12: "a", 13: "a", 25: "a"}), "d3d1"),
        ("empty-throne-pass", fen({60: ".", 61: "k", 55: "a"}), "a6e6"),
        ("king-edge-immunity", fen({60: ".", 5: "k", 4: "a", 6: "a", 27: "a"}), "f3f2"),
        ("king-throne-capture", fen({59: "a", 61: "a", 49: "a", 73: "a"}), "h7f7"),
    ]
    results = []
    with Engine() as referee:
        for name, position, move in fixtures:
            referee.position([], position)
            before = referee.snapshot()
            native = make(
                np.array(
                    [".adk".index(c) for c in unpack(position)], dtype=np.int8
                ).reshape(11, 11),
                position.split()[1],
            )
            reference = set(before["moves"])
            native_legal = legal(native)
            result = dict(
                fixture=name,
                fen=position,
                move=move,
                legal_equal=reference == native_legal,
                missing=sorted(reference - native_legal),
                extra=sorted(native_legal - reference),
            )
            try:
                after = referee.play(move)
                native = apply(native, move)
                expected = np.array(
                    [".adk".index(c) for c in unpack(after["fen"])], dtype=np.int8
                ).reshape(11, 11)
                with contextlib.redirect_stdout(sys.stderr):
                    native_outcome = outcome(native)
                native_board = grid(native).copy()
                raw_equal = bool(np.array_equal(expected, native_board))
                if (
                    after["outcome"]
                    and after["outcome"]["reason"] == "king-captured"
                    and native_outcome == "Attacker"
                ):
                    native_board[native_board == 3] = 0
                reference_outcome = (
                    None
                    if after["outcome"] is None
                    else after["outcome"]["winner"] or "Draw"
                )
                result.update(
                    board_equal=bool(np.array_equal(expected, native_board)),
                    raw_board_equal=raw_equal,
                    terminal_equal=reference_outcome == native_outcome,
                    reference_outcome=after["outcome"],
                    upstream_outcome=native_outcome,
                    upstream_board=grid(native).tolist(),
                )
            except Exception as exc:
                result.update(board_equal=False, error=str(exc))
            results.append(result)
    save(
        args.output,
        dict(
            engine=args.kind,
            scope="focused movement/capture fixtures; not full rules certification",
            compatible_on_fixtures=all(
                r["legal_equal"] and r["board_equal"] and r.get("terminal_equal", False)
                for r in results
            ),
            fixtures=results,
        ),
    )
    print(
        args.kind, [(r["fixture"], r["legal_equal"], r["board_equal"]) for r in results]
    )


if __name__ == "__main__":
    main()
