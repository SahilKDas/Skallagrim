"""Replay completed training games against the current authority; reject divergence."""

import argparse
import json
from pathlib import Path
from transport import Engine
from records import save


def board(fen):
    squares = []
    for rank in fen.split()[0].split("/"):
        digits = ""
        for char in rank + "!":
            if char.isdigit():
                digits += char
                continue
            if digits:
                squares.extend("." * int(digits))
                digits = ""
            if char != "!":
                squares.append(char)
    return squares


def transition_move(before, after):
    old, new = board(before), board(after)
    own = "a" if before.endswith(" a") else "dk"
    starts = [i for i in range(121) if old[i] in own and new[i] == "."]
    ends = [i for i in range(121) if old[i] == "." and new[i] in own]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("ambiguous recorded transition")
    coord = lambda s: f"{chr(97 + s % 11)}{s // 11 + 1}"
    return coord(starts[0]) + coord(ends[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    kept = 0
    rejected = []
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        Engine() as referee,
        args.data.open(encoding="utf8") as src,
        args.output.open("w", encoding="utf8") as dest,
    ):

        def flush(rows):
            nonlocal kept
            if not rows or rows[0].get("target") is None:
                return
            referee.position([])
            state = referee.snapshot()
            try:
                for i, row in enumerate(rows):
                    if row["fen"] != state["fen"] or set(row["legal"]) != set(
                        state["moves"]
                    ):
                        raise ValueError(f"state/legal mismatch at ply {i}")
                    # Next sample's history contains the next FEN, not move; match
                    # the observed transition against all legal moves if needed.
                    # Self-play records now store played_move explicitly.
                    move = row.get("played_move")
                    if move is None:
                        desired = rows[i + 1]["fen"] if i + 1 < len(rows) else None
                        if desired is None:
                            # Find a terminal transition matching the recorded result.
                            desired_outcome = row["outcome"]
                            choices = [r[0] for r in row["policy"]] + state["moves"]
                            move = None
                            for candidate in dict.fromkeys(choices):
                                referee.position([r["_audit_move"] for r in rows[:i]])
                                next_state = referee.play(candidate)
                                if next_state["outcome"] == desired_outcome:
                                    move = candidate
                                    break
                            if move is None:
                                raise ValueError(
                                    f"no matching terminal transition at ply {i}"
                                )
                            referee.position([r["_audit_move"] for r in rows[:i]])
                        else:
                            move = transition_move(row["fen"], desired)
                    row["_audit_move"] = move
                    state = referee.play(move)
                if state["outcome"] != rows[-1]["outcome"]:
                    raise ValueError("outcome mismatch")
            except (ValueError, RuntimeError) as exc:
                rejected.append(dict(game=rows[0]["game"], reason=str(exc)))
                return
            for row in rows:
                row["played_move"] = row.pop("_audit_move")
                row["rules_audited"] = True
                dest.write(json.dumps(row) + "\n")
            kept += 1

        for line in src:
            row = json.loads(line)
            if rows and rows[0]["game"] != row["game"]:
                flush(rows)
                rows = []
            rows.append(row)
        flush(rows)
    save(
        args.output.with_suffix(".audit.json"), dict(kept_games=kept, rejected=rejected)
    )
    print(f"Kept {kept} completed games; rejected {len(rejected)}")


if __name__ == "__main__":
    main()
