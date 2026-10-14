"""One seeded 5,000-training-game round per evaluator, durable and sequential."""

import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from records import save
from transport import ROOT
from league import pin, gate_accepted
from pipeline import validate_checkpoint
from resources import StorageBudget
from campaign_report import figures


def normalized_engines(value, workspace):
    value = json.loads(json.dumps(value))
    for entry in value["engines"].values():

        def normalize(part):
            return str(workspace / part[2:]) if part.startswith("./") else part

        entry["command"] = [
            ";".join(normalize(p) for p in part.split(";")) for part in entry["command"]
        ]
        if entry["command"][0] == "python":
            entry["command"][0] = sys.executable
        for key in ("artifacts", "training_metadata"):
            entry[key] = [
                str(workspace / path) if not Path(path).is_absolute() else path
                for path in entry.get(key, [])
            ]
        for key in ("NNUEFile", "CNNFile"):
            if key in entry.get("options", {}):
                entry["options"][key] = str(workspace / entry["options"][key])
    return value


def freeze(run, workspace):
    runtime = run / "runtime"
    manifest_path = run / "campaign.json"
    if not manifest_path.exists():
        runtime.mkdir(parents=True, exist_ok=True)
        for name in ("skallagrim.exe", "skallagrim-referee.exe"):
            shutil.copy2(workspace / "target/release" / name, runtime / name)
        verification = json.loads(
            (workspace / "benchmarks/search-fix-verification.json").read_text()
        )
        if verification.get("status") != "pass":
            raise ValueError("Search-fix verification must pass before collection")
        for name in ("skallagrim.exe", "skallagrim-referee.exe"):
            if pin(runtime / name)["sha256"] != verification["runtime_sha256"][name]:
                raise ValueError("Runtime differs from the verified search fix")
        for folder in ("python", "models"):
            destination = runtime / folder
            destination.mkdir(exist_ok=True)
            for path in (workspace / folder).glob("*"):
                if path.is_file() and path.suffix in (".py", ".json", ".pt"):
                    shutil.copy2(path, destination / path.name)
        (runtime / "benchmarks").mkdir(exist_ok=True)
        for name in ("league-positions.json", "openings.json"):
            shutil.copy2(workspace / "benchmarks" / name, runtime / "benchmarks" / name)
        engines = normalized_engines(
            json.loads((workspace / "benchmarks/engines.json").read_text()), workspace
        )
        save(runtime / "benchmarks/engines.json", engines)
        for kind in ("nnue", "cnn"):
            validate_checkpoint(
                runtime / "models" / f"{kind}.json", minimum_validation=1
            )
        manifest = dict(
            format="skallagrim-campaign-v1",
            workspace=str(workspace),
            games=6250,
            training_games=5000,
            validation_games=1250,
            generations=1,
            epochs=10,
            batch=64,
            movetime=100,
            cpu=6,
            max_storage_gib=16,
            replay_max_gib=8,
            files={
                path.relative_to(runtime).as_posix(): pin(path)["sha256"]
                for path in runtime.rglob("*")
                if path.is_file()
            },
            seeds={"nnue": 10000, "cnn": 10000},
            created=time.time(),
            source_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
            ).strip()
            if (workspace / ".git").exists()
            else "test fixture",
            search_fix_evidence=pin(
                workspace / "benchmarks/search-fix-verification.json"
            ),
            original_models={
                str(path): pin(path)["sha256"]
                for kind in ("nnue", "cnn")
                for path in [
                    workspace / "models" / f"{kind}{suffix}"
                    for suffix in (".json", ".pt", ".meta.json")
                ]
            },
        )
        # External artifacts stay in place, but their identities are frozen too.
        manifest["external_artifacts"] = {
            path: pin(path)["sha256"]
            for name, entry in engines["engines"].items()
            if name in ("taflzero", "nicholas", "luizz", "opentafl")
            for path in entry.get("artifacts", [])
            if Path(path).is_file()
        }
        save(manifest_path, manifest)
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["files"].items():
        if pin(runtime / name)["sha256"] != digest:
            raise ValueError(f"Frozen campaign input changed: {name}")
    for path, digest in manifest["external_artifacts"].items():
        if pin(path)["sha256"] != digest:
            raise ValueError(f"External artifact changed: {path}")
    if pin(Path(__file__))["sha256"] != manifest["files"]["python/run_campaign.py"]:
        raise ValueError("Controller changed; resume with the frozen controller")
    return runtime, manifest


class Publisher:
    def __init__(self, run, workspace, enabled):
        self.run, self.workspace, self.enabled = run, workspace, enabled
        self.output = workspace / "benchmarks" / run.name
        self.journal = run / "published.json"
        self.published = (
            set(json.loads(self.journal.read_text()))
            if self.journal.exists()
            else set()
        )
        self.events = set()

    def event(self, name, force=False):
        if name in self.published or name in self.events and not force:
            return
        self.events.add(name)
        self.output.mkdir(parents=True, exist_ok=True)
        progress = figures(self.run, self.output / "charts")
        state = (
            json.loads((self.run / "status.json").read_text())
            if (self.run / "status.json").exists()
            else {}
        )
        save(
            self.output / "progress.json",
            dict(
                event=name,
                status=state,
                collection=progress,
                updated=time.time(),
                strength_claim=False,
            ),
        )
        shutil.copy2(self.run / "campaign.json", self.output / "campaign.json")
        artifacts = {}
        runtime = self.run / "runtime"
        for source in [
            runtime / "skallagrim.exe",
            runtime / "skallagrim-referee.exe",
            *list((runtime / "models").glob("*")),
        ]:
            if source.is_file():
                destination = self.output / "seed-runtime" / source.relative_to(runtime)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                artifacts[str(source)] = dict(
                    path=destination.relative_to(self.output).as_posix(),
                    sha256=pin(destination)["sha256"],
                )
        for kind in ("nnue", "cnn"):
            folder = self.run / kind / "generation-0000"
            dest = self.output / kind
            dest.mkdir(exist_ok=True)
            for filename in (
                "complete.json",
                "collection-progress.json",
                "replay-selection.json",
                "split.json",
                f"{kind}.progress.json",
            ):
                if (folder / filename).exists():
                    shutil.copy2(folder / filename, dest / filename)
            checkpoint = folder / f"{kind}.json"
            metadata = checkpoint.with_suffix(".progress.json")
            if metadata.exists():
                value = json.loads(metadata.read_text())
                expected = value.get("model_artifacts", {})
                stable = all(
                    (folder / file).is_file() and pin(folder / file)["sha256"] == sha
                    for file, sha in expected.items()
                )
                if stable and expected or value.get("complete"):
                    for suffix in (".json", ".pt", ".meta.json"):
                        if checkpoint.with_suffix(suffix).exists():
                            shutil.copy2(
                                checkpoint.with_suffix(suffix), dest / f"{kind}{suffix}"
                            )
                            artifacts[str(checkpoint.with_suffix(suffix))] = dict(
                                path=(dest / f"{kind}{suffix}")
                                .relative_to(self.output)
                                .as_posix(),
                                sha256=pin(dest / f"{kind}{suffix}")["sha256"],
                            )
            if (folder / "league.json").exists() and not (
                dest / "league.json.gz"
            ).exists():
                with (
                    (folder / "league.json").open("rb") as src,
                    gzip.open(dest / "league.json.gz", "wb") as out,
                ):
                    shutil.copyfileobj(src, out)
        save(self.output / "artifact-index.json", artifacts)
        if not self.enabled:
            self.published.add(name)
            save(self.journal, sorted(self.published))
            return
        relative = self.output.relative_to(self.workspace).as_posix()
        try:
            staged = subprocess.check_output(
                ["git", "diff", "--cached", "--name-only"],
                cwd=self.workspace,
                text=True,
            ).splitlines()
            if any(not path.startswith(relative + "/") for path in staged):
                raise RuntimeError("Unrelated staged files; publication deferred")
            subprocess.run(
                ["git", "add", "--", relative],
                cwd=self.workspace,
                check=True,
                timeout=60,
            )
            if subprocess.run(
                ["git", "diff", "--cached", "--quiet"], cwd=self.workspace
            ).returncode:
                subprocess.run(
                    [
                        "git",
                        "commit",
                        "-m",
                        f"Trust me bro, here's the actual league progress: {name}",
                    ],
                    cwd=self.workspace,
                    check=True,
                    timeout=60,
                )
            subprocess.run(
                ["git", "push", "origin", "HEAD:main"],
                cwd=self.workspace,
                check=True,
                timeout=120,
            )
            self.published.add(name)
            save(self.journal, sorted(self.published))
        except (subprocess.SubprocessError, RuntimeError) as exc:
            save(
                self.run / "publication-error.json",
                dict(event=name, error=str(exc), updated=time.time()),
            )
            self.events.discard(name)

    def poll(self):
        for kind in ("nnue", "cnn"):
            folder = self.run / kind / "generation-0000"
            path = folder / "collection-progress.json"
            if path.exists():
                progress = json.loads(path.read_text())
                for percent in (25, 50, 75, 100):
                    if progress["completed"] >= (6250 * percent + 99) // 100:
                        self.event(f"{kind}-collection-{percent}%")
                if progress["completed"] >= 50:
                    self.event(f"{kind}-first-50-and-ETA")
            path = folder / f"{kind}.progress.json"
            if path.exists():
                progress = json.loads(path.read_text())
                self.event(f"{kind}-epoch-{progress['epoch']}")
            ready = folder / "evaluation-ready.json"
            ack = folder / "evaluation-ack.json"
            if ready.exists():
                value = json.loads(ready.read_text())
                if not ack.exists() or json.loads(ack.read_text()) != value:
                    self.event(f"{kind}-training-before-evaluation")
                    save(ack, value)


def invoke(command, runtime, budget, publisher, status, name, run):
    configuration = json.loads((run / "campaign.json").read_text())
    for path, expected in configuration["files"].items():
        if pin(runtime / path)["sha256"] != expected:
            raise ValueError("Frozen stage input changed")
    if name == "external-gauntlets":
        for path, expected in configuration["external_artifacts"].items():
            if pin(path)["sha256"] != expected:
                raise ValueError("External opponent changed during training")
    save(
        run / "status.json",
        dict(stage=name, state="running", started=time.time(), command=command),
    )
    env = os.environ | {
        "SKALLAGRIM_ENGINE": str(runtime / "skallagrim.exe"),
        "SKALLAGRIM_REFEREE": str(runtime / "skallagrim-referee.exe"),
        "SKALLAGRIM_CAMPAIGN_BARRIER": "1",
    }
    with (run / (name + ".log")).open("a", encoding="utf8") as log:
        process = subprocess.Popen(
            command, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        save(
            run / "active-stage.json",
            dict(pid=process.pid, running=True, command=command),
        )
        try:
            while process.poll() is None:
                budget.check()
                publisher.poll()
                time.sleep(5)
            if process.returncode:
                raise RuntimeError(
                    f"{name} failed ({process.returncode}); see {name}.log"
                )
        except BaseException:
            if process.poll() is None:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=10,
                )
                process.wait(timeout=10)
            raise
        finally:
            save(
                run / "active-stage.json",
                dict(pid=process.pid, running=process.poll() is None, command=command),
            )
    status.add(name)
    save(run / "completed-stages.json", sorted(status))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=ROOT / "runs/league-5000-2026-10-13")
    p.add_argument("--workspace", type=Path, default=ROOT)
    p.add_argument("--publish", action="store_true")
    args = p.parse_args()
    run, workspace = args.run.resolve(), args.workspace.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock_file = (run / "campaign.lock").open("a+b")
    if lock_file.tell() == 0:
        lock_file.write(b"1")
        lock_file.flush()
    lock_file.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        import ctypes

        previous = run / "active-stage.json"
        if previous.exists() and json.loads(previous.read_text()).get("running"):
            pid = json.loads(previous.read_text())["pid"]
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.GetExitCodeProcess.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_ulong),
            ]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel.OpenProcess(0x1000, 0, pid)
            if handle:
                code = ctypes.c_ulong()
                kernel.GetExitCodeProcess(handle, ctypes.byref(code))
                kernel.CloseHandle(handle)
                if code.value == 259:
                    raise RuntimeError(
                        f"Previous owned stage PID {pid} is still active; refusing a duplicate heavy job"
                    )
    # Interrupted atomic writes are not committed game/checkpoint data.
    for temporary in run.rglob("*.tmp"):
        if (
            temporary.is_file()
            and not temporary.is_symlink()
            and temporary.resolve().is_relative_to(run)
        ):
            temporary.unlink()
    # Search supervisor thread boosts only its response; all heavy work remains idle.
    if os.name == "nt":
        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = kernel.GetCurrentProcess()
        if not kernel.SetPriorityClass(
            handle, 0x40
        ) or not kernel.SetProcessAffinityMask(handle, 1 << 6):
            raise ctypes.WinError()
    budget = StorageBudget(run, 16 * 1024**3)
    budget.check()
    runtime, configuration = freeze(run, workspace)
    publisher = Publisher(run, workspace, args.publish)
    done_path = run / "completed-stages.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    save(
        run / "worker.json",
        dict(pid=os.getpid(), started=time.time(), publication=args.publish),
    )
    try:
        if "seed-parity" not in done:
            invoke(
                [
                    sys.executable,
                    "python/verify_models.py",
                    "--models",
                    str(runtime / "models"),
                    "--positions",
                    "80",
                ],
                runtime,
                budget,
                publisher,
                done,
                "seed-parity",
                run,
            )
        publisher.event("campaign-started")
        for kind in ("nnue", "cnn"):
            if kind not in done:
                invoke(
                    [
                        sys.executable,
                        "python/pipeline.py",
                        "--kind",
                        kind,
                        "--generations",
                        "1",
                        "--games",
                        "6250",
                        "--training-games",
                        "5000",
                        "--validation-games",
                        "1250",
                        "--movetime",
                        "100",
                        "--epochs",
                        "10",
                        "--batch",
                        "64",
                        "--window-games",
                        "6250",
                        "--initial-champion",
                        str(runtime / "models" / f"{kind}.json"),
                        "--league-positions",
                        str(runtime / "benchmarks/league-positions.json"),
                        "--run",
                        str(run / kind),
                        "--storage-root",
                        str(run),
                        "--max-storage-gib",
                        "16",
                        "--replay-max-gib",
                        "8",
                        "--cpu",
                        "6",
                    ],
                    runtime,
                    budget,
                    publisher,
                    done,
                    kind,
                    run,
                )
            publisher.event(f"{kind}-training-and-promotion-results")
        config = json.loads((runtime / "benchmarks/engines.json").read_text())
        for bot, kind in (("rho", "cnn"), ("iota", "nnue"), ("beta", "nnue")):
            model = run / kind / "generation-0000" / f"{kind}.json"
            validate_checkpoint(model, 5000, 1250)
            entry = config["engines"][bot]
            entry["command"] = [str(runtime / "skallagrim.exe")]
            entry["options"]["CNNFile" if kind == "cnn" else "NNUEFile"] = str(model)
            entry["artifacts"] = [
                str(runtime / "skallagrim.exe"),
                str(model),
                str(model.with_suffix(".meta.json")),
            ]
            entry["training_metadata"] = [str(model.with_suffix(".meta.json"))]
        config_path = run / "candidate-engines.json"
        save(config_path, config)
        if "external-gauntlets" not in done:
            invoke(
                [
                    sys.executable,
                    "python/gauntlet_suite.py",
                    "--config",
                    str(config_path),
                    "--output",
                    str(run / "external"),
                    "--resume",
                    "--allow-unaudited",
                    "--load-note",
                    "Idle-priority campaign; one logical CPU 6; ambient desktop load recorded, uncontrolled",
                ],
                runtime,
                budget,
                publisher,
                done,
                "external-gauntlets",
                run,
            )
        if "replay-audit" not in done:
            invoke(
                [
                    sys.executable,
                    "python/audit_matches.py",
                    "--root",
                    str(run),
                    "--output",
                    str(run / "replay-audit.json"),
                    "--verify-artifacts",
                ],
                runtime,
                budget,
                publisher,
                done,
                "replay-audit",
                run,
            )
        deployment = {}
        for kind in ("nnue", "cnn"):
            complete = json.loads(
                (run / kind / "generation-0000/complete.json").read_text()
            )
            required = {
                f"{bot}/{name}"
                for bot in (["rho"] if kind == "cnn" else ["iota", "beta"])
                for name in ["champion", "held-out"]
            }
            gates = complete.get("gate") or {}
            verified_acceptance = required.issubset(gates) and gate_accepted(
                gates, {key for key in gates if key.endswith("/champion")}
            )
            if bool(complete["accepted"]) != verified_acceptance:
                raise ValueError(
                    "Saved promotion decision contradicts the role/confidence gates"
                )
            deployment[kind] = verified_acceptance
            if complete["accepted"]:
                source = run / kind / "generation-0000" / f"{kind}.json"
                for suffix in (".pt", ".meta.json", ".json"):
                    destination = workspace / "models" / f"{kind}{suffix}"
                    expected = configuration["original_models"][str(destination)]
                    candidate_hash = pin(source.with_suffix(suffix))["sha256"]
                    if pin(destination)["sha256"] not in (expected, candidate_hash):
                        raise ValueError(
                            "Current champion changed externally; refusing deployment"
                        )
                for suffix in (".pt", ".meta.json", ".json"):
                    destination = workspace / "models" / f"{kind}{suffix}"
                    temporary = destination.with_suffix(
                        destination.suffix + ".campaign.tmp"
                    )
                    shutil.copy2(source.with_suffix(suffix), temporary)
                    temporary.replace(destination)
        save(
            run / "status.json",
            dict(
                stage="complete",
                state="complete",
                promoted=deployment,
                stronger_claim={
                    kind: accepted for kind, accepted in deployment.items()
                },
                completed=time.time(),
            ),
        )
        # Full match replays already include referee evidence; publish compressed copies.
        for source in run.rglob("*-??.json"):
            relative = source.relative_to(run)
            destination = (
                publisher.output / "matches" / relative.with_suffix(".json.gz")
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, gzip.open(destination, "wb") as out:
                shutil.copyfileobj(src, out)
        shutil.copy2(run / "replay-audit.json", publisher.output / "replay-audit.json")
        publisher.event("final-gauntlets-and-promotion", force=True)
        if deployment.get("nnue") or deployment.get("cnn"):
            # Deployment publication is explicit and scoped to promoted model files.
            if args.publish:
                paths = [
                    f"models/{kind}{suffix}"
                    for kind, accepted in deployment.items()
                    if accepted
                    for suffix in (".json", ".pt", ".meta.json")
                ]
                staged = subprocess.check_output(
                    ["git", "diff", "--cached", "--name-only"], cwd=workspace, text=True
                ).splitlines()
                if any(path not in paths for path in staged):
                    raise RuntimeError(
                        "Unrelated staged files; deployment publication deferred"
                    )
                subprocess.run(["git", "add", "--", *paths], cwd=workspace, check=True)
                if subprocess.run(
                    ["git", "diff", "--cached", "--quiet"], cwd=workspace
                ).returncode:
                    subprocess.run(
                        [
                            "git",
                            "commit",
                            "-m",
                            "Trust me bro, this champion actually passed the gates",
                        ],
                        cwd=workspace,
                        check=True,
                    )
                subprocess.run(
                    ["git", "push", "origin", "HEAD:main"], cwd=workspace, check=True
                )
        if args.publish and "final-gauntlets-and-promotion" not in publisher.published:
            raise RuntimeError(
                "Final GitHub publication is pending; local results are durable"
            )
    except BaseException as exc:
        save(
            run / "status.json",
            dict(
                stage="attention",
                state="failed",
                error=repr(exc),
                updated=time.time(),
                resumable=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
