"""Pin earned campaign champions for subsequent search-only experiments.

Does not launch searches, alter the frozen campaign, or promote candidates.
"""

import argparse
import hashlib
import json
from pathlib import Path

from league import gate_accepted


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pin(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_baseline(run):
    run = Path(run).resolve()
    status = read(run / "status.json")
    if status.get("state") != "complete" or status.get("stage") != "complete":
        raise ValueError("Campaign must finish before establishing the baseline")
    done = set(read(run / "completed-stages.json"))
    if not {"nnue", "cnn", "external-gauntlets", "replay-audit"}.issubset(done):
        raise ValueError("Campaign stages are incomplete")
    audit = read(run / "replay-audit.json")
    if not audit.get("replay_verified") or audit.get("games", 0) < 750:
        raise ValueError("Final comparison replay audit is incomplete")
    runtime = run / "runtime"
    manifest = read(run / "campaign.json")
    for relative, expected in manifest["files"].items():
        if pin(runtime / relative)["sha256"] != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    config = read(runtime / "benchmarks/engines.json")
    results = {}
    for kind, bots in [("nnue", ["iota", "beta"]), ("cnn", ["rho"])]:
        folder = run / kind / "generation-0000"
        complete = read(folder / "complete.json")
        gates = complete.get("gate") or {}
        required = {
            f"{bot}/{opponent}" for bot in bots for opponent in ["champion", "held-out"]
        }
        accepted = required.issubset(gates) and gate_accepted(
            gates, {key for key in gates if key.endswith("/champion")}
        )
        if (
            complete.get("accepted") != accepted
            or status["promoted"].get(kind) != accepted
        ):
            raise ValueError(f"Contradictory promotion decision: {kind}")
        model = (folder if accepted else runtime / "models") / f"{kind}.json"
        metadata = read(model.with_suffix(".meta.json"))
        artifacts = {
            model.with_suffix(suffix).name: pin(model.with_suffix(suffix))
            for suffix in [".json", ".pt", ".meta.json"]
        }
        for name, expected in metadata.get("model_artifacts", {}).items():
            if name in artifacts and artifacts[name]["sha256"] != expected:
                raise ValueError(f"Checkpoint artifact changed: {name}")
        results[kind] = {
            "promoted": accepted,
            "artifacts": artifacts,
            "decision": pin(folder / "complete.json"),
        }
        for bot in bots:
            entry = config["engines"][bot]
            entry["command"] = [str(runtime / "skallagrim.exe")]
            entry["options"]["CNNFile" if kind == "cnn" else "NNUEFile"] = str(model)
            entry["artifacts"] = [
                str(runtime / "skallagrim.exe"),
                str(model),
                str(model.with_suffix(".meta.json")),
            ]
            entry["training_metadata"] = [str(model.with_suffix(".meta.json"))]
    return {
        "format": "skallagrim-search-baseline-v1",
        "campaign": str(run),
        "engine": pin(runtime / "skallagrim.exe"),
        "champions": results,
        "replay_audit": pin(run / "replay-audit.json"),
        "clocks": {"movetime_ms": 300, "ipc_allowance_ms": 25},
        "strength_claim": "Promotion decisions only; retain published timing and upstream limitations",
        "config": config,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_baseline(args.run)
    if args.output.exists():
        if read(args.output) != result:
            raise ValueError("Refusing to replace a different pinned baseline")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
