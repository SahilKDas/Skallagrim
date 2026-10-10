"""Replay archived gauntlets and verify moves, final boards and recorded scores."""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
from records import save
from transport import Engine, ROOT
from referee import Referee

FAILURES = {"timeout", "illegal-move", "engine-failure", "rules-incompatibility"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Archived artifacts mirroring their original relative paths",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        help="Exact archived referee binary for historical rule versions",
    )
    args = parser.parse_args()
    checked = 0
    plies = 0
    artifacts = {}
    if args.verify_artifacts:
        for path in args.root.rglob("manifest.json"):
            manifest = json.loads(path.read_text(encoding="utf8"))
            entries = list(manifest["engines"].values())
            if manifest.get("referee"):
                entries.append(manifest["referee"])
            for entry in entries:
                for name, digest in entry["sha256"].items():
                    if name in artifacts:
                        assert artifacts[name] == digest, (
                            name,
                            "conflicting artifact hashes",
                        )
                    artifacts[name] = digest
        for name, expected in artifacts.items():
            path = (
                args.artifact_root / name
                if args.artifact_root and (args.artifact_root / name).is_file()
                else ROOT / name
            )
            with path.open("rb") as file:
                assert hashlib.file_digest(file, "sha256").hexdigest() == expected, (
                    name,
                    "artifact changed",
                )
    with contextlib.ExitStack() as lifecycle:
        referee = lifecycle.enter_context(
            Engine([str(args.engine.resolve())] if args.engine else None)
        )
        audit = None
        for path in sorted(args.root.rglob("*-??.json")):
            game = json.loads(path.read_text(encoding="utf8"))
            if game.get("referee"):
                if audit is None:
                    audit = lifecycle.enter_context(Referee())
                audit.call("validate", replay=game)
            referee.position(game["opening"]["moves"], game.get("start"))
            state = referee.snapshot()
            for entry in game["moves"]:
                assert entry["side"] == state["side"], (path, "side")
                assert entry["move"] in state["moves"], (path, "illegal move")
                assert not state["outcome"], (path, "move after terminal state")
                state = referee.play(entry["move"])
                plies += 1
            assert state["fen"] == game["final_fen"], (path, "final board")
            expected = (
                0.5
                if game["winner"] is None
                else int(game["winner"] == game["candidate_side"])
            )
            assert game["score"] == expected, (path, "score")
            if game["reason"] in FAILURES:
                assert not state["outcome"], (path, "forfeit after terminal state")
                assert game["winner"] != state["side"], (path, "forfeit winner")
            elif game["reason"] == "move-limit-adjudication":
                assert not state["outcome"] and game["winner"] is None, path
            elif game["reason"] == "exit-fort-dynamic":
                assert game.get("referee", {}).get("version") == 2 and game["winner"] == "Defender", path
            else:
                assert state["outcome"] == {
                    "winner": game["winner"],
                    "reason": game["reason"],
                }, (path, "outcome")
            checked += 1
    save(
        args.output,
        dict(
            games=checked,
            plies=plies,
            replay_verified=True,
            independent_rules_certification=False,
            verified_artifacts=len(artifacts),
        ),
    )
    print(f"Verified {checked} games and {plies} plies")


if __name__ == "__main__":
    main()
