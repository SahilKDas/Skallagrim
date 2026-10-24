"""Isolated, resumable evaluator A/B matches and a 2,000-game corpus extension.

One child stage at a time; no promotion, Git writes, or edits to source corpora.
The runtime is frozen before launch so later working-tree edits cannot affect it.
"""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from dataset import DatasetIndex, make_replay_sqlite
from league import gate_accepted, pin
from pin_campaign_baseline import build_baseline
from records import save
from resources import StorageBudget
from transport import ROOT, pin_current_process


def extension_split(old, index, extension, seed):
    """Keep every old partition assignment and stratify only the new games."""
    training, validation = set(old["training"]), set(old["validation"])
    if training & validation or (training | validation) & set(extension.groups):
        raise ValueError("Old split overlaps or extension reuses old game IDs")
    rng = random.Random(seed)
    for role in ("Attacker", "Defender"):
        games = sorted(g for g in extension.groups if extension.roles.get(g) == role)
        if len(games) != 500:
            raise ValueError("Extension needs exactly 500 games per candidate role")
        rng.shuffle(games)
        validation.update(games[:100])
        training.update(games[100:])
    if training | validation != set(index.groups):
        raise ValueError("Combined replay did not preserve the full old corpus")
    return dict(format="skallagrim-split-v1", data_sha256=index.sha256,
                seed=seed, training=sorted(training), validation=sorted(validation))


def prepare(run, source, baseline_path):
    path = run / "campaign.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["source_campaign"] != str(source):
            raise ValueError("Changed source campaign")
        verify_inputs(manifest)
        return manifest
    baseline = build_baseline(source)
    if json.loads(baseline_path.read_text()) != baseline:
        raise ValueError("Baseline does not match earned campaign champions")
    runtime = run / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for name in ("skallagrim.exe", "skallagrim-referee.exe"):
        shutil.copy2(ROOT / "target/release" / name, runtime / name)
    for folder, pattern in [("python", "*.py"), ("src", "*.rs"), ("tests", "*.rs")]:
        for item in (ROOT / folder).rglob(pattern):
            if "__pycache__" in item.parts:
                continue
            destination = runtime / item.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
    for name in ("Cargo.toml", "Cargo.lock"):
        shutil.copy2(ROOT / name, runtime / name)
    (runtime / "benchmarks").mkdir(exist_ok=True)
    for name in ("openings.json", "league-positions.json"):
        shutil.copy2(ROOT / "benchmarks" / name, runtime / "benchmarks" / name)
    inputs = {}
    for kind in ("nnue", "cnn"):
        for artifact in baseline["champions"][kind]["artifacts"].values():
            item = Path(artifact["path"])
            if pin(item) != artifact:
                raise ValueError("Champion artifact changed")
            destination = runtime / "models" / item.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(item, destination)
        for name in ("replay.sqlite", "split.json"):
            item = source / kind / "generation-0000" / name
            inputs[f"{kind}/{name}"] = pin(item)
    manifest = dict(format="skallagrim-alternate-extension-v1",
        source_campaign=str(source), baseline=pin(baseline_path), inputs=inputs,
        files={p.relative_to(runtime).as_posix(): pin(p) for p in runtime.rglob("*") if p.is_file()},
        created=time.time(), deadline=time.time() + 48 * 3600,
        additional_completed_games=2000, per_kind=1000, new_training_per_kind=800,
        new_validation_per_kind=200, epochs=10, cpu=6,
        collection_movetime_ms=100, comparison_movetime_ms=300, ipc_allowance_ms=25,
        comparison_games_per_bot=50, blend_percent=25, max_run_bytes=2_800_000_000,
        automatic_promotion=False, collection_blend_percent=0,
        comparison="Same frozen binary and champion weights; only HandcraftedWeight changes",
        limitations="50 games per bot are a screening test, not definitive strength proof")
    save(path, manifest)
    save(run / "status.json", dict(state="prepared", additional_completed_games=0))
    return manifest


def verify_inputs(manifest):
    for artifact in [manifest["baseline"], *manifest["inputs"].values(), *manifest["files"].values()]:
        if pin(Path(artifact["path"])) != artifact:
            raise ValueError(f"Frozen input changed: {artifact['path']}")


def active_process(pid):
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, 0, pid)
        if not handle:
            if ctypes.get_last_error() == 5:
                raise RuntimeError("Cannot verify previous child; refusing duplicate work")
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def engine_entry(runtime, bot, kind, weight, model=None):
    model = model or runtime / "models" / f"{kind}.json"
    return dict(command=[str(runtime / "skallagrim.exe")],
        options={"Bot": bot, "CNNFile" if kind == "cnn" else "NNUEFile": str(model),
                 "HandcraftedWeight": weight},
        artifacts=[str(runtime / "skallagrim.exe"), str(model), str(model.with_suffix(".meta.json"))],
        training_metadata=[str(model.with_suffix(".meta.json"))], rules_audited=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run, source = args.run.resolve(), args.source_campaign.resolve()
    if run == source or run.is_relative_to(source) or source.is_relative_to(run):
        raise ValueError("Alternate run must be separate from the frozen campaign")
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "campaign.lock").open("a+b")
    if lock.tell() == 0:
        lock.write(b"1"); lock.flush()
    lock.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = run / "status.json"
    if previous.exists():
        prior = json.loads(previous.read_text())
        if prior.get("state") == "running" and prior.get("child_pid") and active_process(prior["child_pid"]):
            raise RuntimeError(f"Previous child PID {prior['child_pid']} is still active; refusing duplicate work")
    manifest = prepare(run, source, args.baseline.resolve())
    if args.prepare_only:
        return
    pin_current_process(manifest["cpu"])
    runtime = run / "runtime"
    budget = StorageBudget(run, manifest["max_run_bytes"])
    env = os.environ | {"SKALLAGRIM_ENGINE": str(runtime / "skallagrim.exe"),
        "SKALLAGRIM_REFEREE": str(runtime / "skallagrim-referee.exe"),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    done_path = run / "completed-stages.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    started = time.monotonic()
    monotonic_deadline = started + max(0, manifest["deadline"] - time.time())

    def check():
        budget.check()
        if budget.available_bytes() < 32 * 1024**2:
            raise RuntimeError("Storage headroom exhausted; committed artifacts retained")
        if time.time() >= manifest["deadline"] or time.monotonic() >= monotonic_deadline:
            raise RuntimeError("48-hour campaign bound reached; committed stages retained")

    def status(stage, state, **extra):
        progress = {}
        for kind in ("nnue", "cnn"):
            path = run / "extension" / kind / "collection-progress.json"
            if path.exists(): progress[kind] = json.loads(path.read_text())
        save(run / "status.json", dict(state=state, stage=stage, pid=os.getpid(),
            updated=time.time(), completed_stages=sorted(done), collection=progress,
            additional_completed_games=sum(p["completed"] for p in progress.values()), **extra))

    def invoke(stage, script, *arguments):
        if stage in done: return
        check()
        verify_inputs(manifest)
        command = [sys.executable, "-u", str(runtime / "python" / script), *map(str, arguments)]
        with (run / f"{stage}.log").open("a", encoding="utf8") as log:
            process = subprocess.Popen(command, cwd=runtime, env=env,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                while process.poll() is None:
                    check()
                    status(stage, "running", child_pid=process.pid, command=command)
                    time.sleep(5)
                if process.returncode:
                    raise RuntimeError(f"{stage} failed ({process.returncode}); inspect {stage}.log")
            finally:
                if process.poll() is None:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            capture_output=True, timeout=10)
                    else: process.terminate()
                    process.wait(timeout=10)
        done.add(stage)
        save(done_path, sorted(done))

    def comparison(bot, kind, candidate_model=None):
        label = "training" if candidate_model else "blend"
        folder = run / "comparisons" / label / bot.lower()
        folder.mkdir(parents=True, exist_ok=True)
        config = folder / "engines.json"
        save(config, dict(engines={
            "candidate": engine_entry(runtime, bot, kind, 0 if candidate_model else 25, candidate_model),
            "control": engine_entry(runtime, bot, kind, 0)}))
        invoke(f"{label}-{bot.lower()}", "benchmark.py", "--config", config,
            "--candidate", "candidate", "--opponents", "control", "--output", folder,
            "--games", 50, "--movetime", 300, "--ipc-allowance", 25,
            "--max-plies", 600, "--cpu", 6, "--resume",
            "--load-note", "Sequential Idle priority on CPU 6; ambient load uncontrolled")
        invoke(f"audit-{label}-{bot.lower()}", "audit_matches.py", "--root", folder,
            "--output", folder / "audit.json", "--verify-artifacts")
        summary = json.loads((folder / "control-summary.json").read_text())
        assessment = dict(summary=summary,
            passes_screen=gate_accepted({bot: summary}, {bot}),
            automatic_promotion=False, candidate_handcrafted_percent=0 if candidate_model else 25,
            control_handcrafted_percent=0, evaluation="same bot, paired starts and swapped roles")
        save(folder / "assessment.json", assessment)

    try:
        invoke("parity", "verify_models.py", "--models", runtime / "models", "--positions", 40)
        # Collection starts first; evaluator matches follow. The controls use
        # frozen champion weights, so training cannot confound the blend test.
        for i, (kind, bot) in enumerate((("nnue", "Beta"), ("cnn", "Rho"))):
            folder = run / "extension" / kind
            folder.mkdir(parents=True, exist_ok=True)
            league_path = folder / "league.json"
            if not league_path.exists():
                # Build with the frozen runtime modules and explicit paths.
                code = ("from league import manifest,checkpoint; from pathlib import Path; "
                    f"p=Path({str(runtime)!r}); c=checkpoint(p/'models/{kind}.json',{bot!r}); "
                    f"manifest(Path({str(league_path)!r}),c,c,[],p/'benchmarks/league-positions.json',"
                    f"1000,10000,{2026102101+i},100,600)")
                subprocess.run([sys.executable, "-c", code], cwd=runtime / "python", env=env,
                               check=True, timeout=60)
            invoke(f"collect-{kind}", "selfplay.py", "--league", league_path,
                "--bot", bot, "--games", 1000, "--max-attempts", 10000,
                "--seed", 2026102101 + i, "--movetime", 100, "--max-plies", 600,
                "--output", folder / "selfplay.jsonl", "--resume-collection", "--cpu", 6,
                "--storage-root", run, "--max-storage-gib", manifest["max_run_bytes"] / 1024**3)
            prepare_stage = f"prepare-{kind}"
            combined = folder / "replay.sqlite"
            split_path = folder / "split.json"
            if prepare_stage not in done:
                status(prepare_stage, "running")
                check()
                old_data = Path(manifest["inputs"][f"{kind}/replay.sqlite"]["path"])
                old_split = json.loads(Path(manifest["inputs"][f"{kind}/split.json"]["path"]).read_text())
                old_index = DatasetIndex(old_data)
                if old_index.sha256 != old_split["data_sha256"]:
                    raise ValueError("Old replay differs from its frozen split")
                new_data = folder / "selfplay.jsonl.gz"
                extension = DatasetIndex(new_data)
                if len(extension.groups) != 1000:
                    raise ValueError("Extension did not produce exactly 1,000 completed games")
                selection = make_replay_sqlite([old_data, new_data], combined,
                    len(old_index.groups) + 1000, min(1_000_000_000, budget.available_bytes()),
                    len(old_index.groups) + 1000, check=check)
                split = extension_split(old_split, DatasetIndex(combined), extension, 20261021)
                save(split_path, split)
                save(folder / "replay-selection.json", selection)
                done.add(prepare_stage); save(done_path, sorted(done))
            invoke(f"train-{kind}", "train.py", "--kind", kind, "--data", combined,
                "--split", split_path, "--resume", runtime / "models" / f"{kind}.pt",
                "--output", folder / f"{kind}.json", "--epoch-state", folder / "epoch-state.pt",
                "--epochs", 10, "--batch", 64, "--device", "cpu",
                "--learning-rate", 0.0003, "--deadline-unix", manifest["deadline"])
        parity = run / "candidate-models"
        parity.mkdir(exist_ok=True)
        for kind in ("nnue", "cnn"):
            for suffix in (".json", ".pt", ".meta.json"):
                shutil.copy2(run / "extension" / kind / f"{kind}{suffix}", parity / f"{kind}{suffix}")
        invoke("candidate-parity", "verify_models.py", "--models", parity, "--positions", 80)
        for bot, kind in (("Iota", "nnue"), ("Beta", "nnue"), ("Rho", "cnn")):
            comparison(bot, kind)
            comparison(bot, kind, run / "extension" / kind / f"{kind}.json")
        verify_inputs(manifest)
        results = {str(p.parent.relative_to(run)): json.loads(p.read_text())
                   for p in run.glob("comparisons/*/*/assessment.json")}
        save(run / "results.json", dict(additional_completed_games=2000,
            comparisons=results, automatic_promotion=False))
        status("complete", "complete")
    except BaseException as exc:
        status("interrupted", "failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
