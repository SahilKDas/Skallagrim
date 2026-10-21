"""Run a single resumable, one-hour NNUE study after the audited campaign."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pin_campaign_baseline import build_baseline, pin
from records import save
from resources import StorageBudget
from transport import pin_current_process


CONFIGURATIONS = [
    ("control", []),
    ("lower-lr", ["--learning-rate", "0.0003"]),
    ("regularized", ["--learning-rate", "0.0003", "--weight-decay", "0.05"]),
    ("symmetry", ["--learning-rate", "0.0003", "--augment"]),
    (
        "features-v2",
        ["--learning-rate", "0.0003", "--feature-version", "v2", "--migrate-nnue"],
    ),
    ("hidden-128", ["--learning-rate", "0.0003", "--hidden", "128", "--migrate-nnue"]),
]


def study_state(manifest, identity, now):
    if manifest.exists():
        study = json.loads(manifest.read_text())
        if study["identity"] != identity:
            raise ValueError("Study inputs changed")
        return study
    study = dict(
        identity=identity, started=now, deadline=now + 3600, results=[], complete=False
    )
    save(manifest, study)
    return study


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--verification", required=True, type=Path)
    args = p.parse_args()
    verification = json.loads(args.verification.read_text())
    if verification.get("passed") is not True:
        raise ValueError("Tooling verification has not passed")
    for artifact in verification["source_files"].values():
        if pin(Path(artifact["path"])) != artifact:
            raise ValueError("Tooling changed since verification")
    run, output = args.run.resolve(), args.output.resolve()
    if not output.is_relative_to(run.parent) or output.is_relative_to(run):
        raise ValueError(
            "Experiments must be outside frozen run but inside shared storage root"
        )
    baseline = build_baseline(run)
    budget = StorageBudget(run.parent, 16 * 1024**3)
    budget.check()
    pin_current_process()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "study.lock"
    # Exclusive creation prevents duplicate runs. An interrupted coordinator's
    # lock must be reviewed against its recorded PID before manual recovery.
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    try:
        manifest = output / "study.json"
        data = run / "nnue/generation-0000/replay.sqlite"
        split = run / "nnue/generation-0000/split.json"
        checkpoint = Path(baseline["champions"]["nnue"]["artifacts"]["nnue.pt"]["path"])
        identity = dict(
            baseline=baseline,
            data=pin(data),
            split=pin(split),
            configurations=CONFIGURATIONS,
            seconds=3600,
        )
        # Normalize tuples for comparisons after JSON round trips.
        identity = json.loads(json.dumps(identity))
        study = study_state(manifest, identity, time.time())
        monotonic_deadline = time.monotonic() + min(
            3600, max(0, study["deadline"] - time.time())
        )
        for name, options in CONFIGURATIONS:
            if any(row["name"] == name for row in study["results"]):
                continue
            remaining = min(
                study["deadline"] - time.time(), monotonic_deadline - time.monotonic()
            )
            if remaining <= 0:
                study["results"].append(
                    dict(name=name, state="unstarted-budget-exhausted")
                )
                continue
            budget.check()
            folder = output / name
            folder.mkdir(exist_ok=True)
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("train.py")),
                "--kind",
                "nnue",
                "--data",
                str(data),
                "--split",
                str(split),
                "--resume",
                str(checkpoint),
                "--output",
                str(folder / "nnue.json"),
                "--epoch-state",
                str(folder / "epoch-state.pt"),
                "--epochs",
                "10",
                "--patience",
                "2",
                "--training-games",
                "5000",
                "--validation-games",
                "1250",
                "--deadline-unix",
                str(study["deadline"]),
                *options,
            ]
            with (folder / "training.log").open("a") as log:
                proc = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    | getattr(subprocess, "IDLE_PRIORITY_CLASS", 0),
                )
                try:
                    code = proc.wait(
                        timeout=max(
                            0.001,
                            min(
                                study["deadline"] - time.time(),
                                monotonic_deadline - time.monotonic(),
                            ),
                        )
                    )
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    code = 75
            study["results"].append(
                dict(
                    name=name,
                    state="complete"
                    if code == 0
                    else "incomplete-budget"
                    if code == 75
                    else "failed",
                    returncode=code,
                    checkpoint_present=(folder / "epoch-state.pt").exists(),
                )
            )
            save(manifest, study)
            if code not in [0, 75]:
                raise RuntimeError(f"Experiment {name} failed; inspect training.log")
        study.update(
            complete=True,
            finished=time.time(),
            automatic_promotion=False,
            label_mode="outcomes; mixed targets deferred until separately verified labels exist",
        )
        save(manifest, study)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
