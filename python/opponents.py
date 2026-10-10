"""Fetch pinned opponents and report prerequisites without inventing weights."""

import argparse
import json
import subprocess
from transport import ROOT
from records import save

OPPONENTS = {
    "taflzero": (
        "https://github.com/sovelin/taflzero",
        "9b5a19a48a6c10cc27d14bd97859faa1f2953120",
    ),
    "nicholas": (
        "https://github.com/nicholasg1997/hnefatafl",
        "1fca52c52233115d4cbb01603bf42bc8d0a9dc6d",
    ),
    "luizz": (
        "https://github.com/LuizzK/hnefatafl-engine",
        "02e9b281200b4d9a3ced463ddcf44be527492aba",
    ),
    "opentafl": (
        "https://github.com/jslater89/OpenTafl",
        "6895051d5a3b32612d12e93e2cd4584d23f94bb3",
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true")
    args = p.parse_args()
    records = {}
    for name, (url, revision) in OPPONENTS.items():
        path = ROOT / "external" / name
        if args.fetch:
            if not path.exists():
                subprocess.run(
                    ["git", "clone", "--no-checkout", url, str(path)], check=True
                )
            current = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True
            ).stdout.strip()
            if current != revision:
                dirty = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=path,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                if dirty:
                    raise SystemExit(
                        f"Refusing to change dirty opponent checkout: {path}"
                    )
                subprocess.run(
                    ["git", "fetch", "origin", revision], cwd=path, check=True
                )
                subprocess.run(
                    ["git", "checkout", "--detach", revision], cwd=path, check=True
                )
        present = path.exists()
        actual = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True
            ).stdout.strip()
            if present
            else None
        )
        weights = (
            list(path.rglob("*.onnx"))
            + list(path.rglob("*.pt"))
            + list(path.rglob("*.ckpt"))
            if present
            else []
        )
        records[name] = dict(
            url=url,
            required_revision=revision,
            actual_revision=actual,
            pinned=actual == revision,
            weights=[str(w.relative_to(ROOT)) for w in weights],
            weight_provenance={
                str(w.relative_to(ROOT)): json.loads(
                    w.with_suffix(".meta.json").read_text()
                ).get("provenance", "local checkpoint")
                if w.with_suffix(".meta.json").exists()
                else "repository-provided artifact"
                for w in weights
            },
            rules_audit="focused audit incompatible"
            if (ROOT / f"benchmarks/{name}-rules-audit.json").exists()
            and not json.loads(
                (ROOT / f"benchmarks/{name}-rules-audit.json").read_text()
            )["compatible_on_fixtures"]
            else "full audit pending",
        )
    save(ROOT / "benchmarks/opponents.lock.json", records)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
