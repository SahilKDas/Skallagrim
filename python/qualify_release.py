"""Clean-directory process/render soak; never claims native interaction coverage."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

from transport import Engine
from referee import Referee


class Memory(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
        (name, ctypes.c_size_t)
        for name in ("peak", "working", "paged_peak", "paged", "nonpaged_peak", "nonpaged", "pagefile", "pagefile_peak")
    ]


def memory(process):
    counters = Memory()
    counters.cb = ctypes.sizeof(counters)
    fn = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not fn(int(process._handle), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.working


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=7200)
    args = p.parse_args()
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.archive) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert (root / name).resolve().is_relative_to(root)
        archive.extractall(root)
    checksums = json.loads((root / "checksums.json").read_text())
    for name, digest in checksums.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    for name in list(os.environ):
        if name.startswith("SKALLAGRIM_"):
            del os.environ[name]
    os.chdir(root)
    started = time.monotonic()
    report = dict(version="0.5.0-beta.1", status="running", payloads_verified=len(checksums),
                  archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                  requested_seconds=args.seconds, batches=0, plies=0, cancellations=0,
                  replay_audits=0, render_checks=0, memory_samples=[], failures=[],
                  deadline_fallbacks={bot: 0 for bot in ("Iota", "Beta", "Rho")},
                  native_gui_checks="untested: native control unavailable",
                  full_training="not run", strength_gauntlets="not run", live_discord="not run: no application ID")
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        args.report.write_text(json.dumps(report, indent=2))

    engines = []
    ref = None
    context = {"operation": "startup"}
    try:
        ref = Referee(root / "skallagrim-referee.exe", root / "soak logs")
        for bot in ("Iota", "Beta", "Rho"):
            engines.append(Engine([str(root / "skallagrim.exe")], options={
                "Bot": bot, "NNUEFile": str(root / "models/nnue.json"),
                "CNNFile": str(root / "models/cnn.json")}, cwd=root))
            engines[-1].send("uci")
            _, lines = engines[-1].until("uciok")
            assert "id name Skallagrim 0.5.0-beta.1" in lines
        while time.monotonic() - started < args.seconds:
            moves = []
            state = ref.position()
            for ply in range(12):
                if state.get("outcome"):
                    break
                engine = engines[(report["batches"] + ply) % 3]
                context = dict(operation="search", batch=report["batches"], ply=ply,
                               bot=("Iota", "Beta", "Rho")[(report["batches"] + ply) % 3],
                               fen=state["fen"], moves=list(moves))
                engine.position(moves)
                if ply == 0:
                    context["operation"] = "cancellation"
                    engine.send("go movetime 1000")
                    engine.send("stop")
                    engine.until("bestmove", 5)
                    engine.ready()
                    report["cancellations"] += 1
                context["operation"] = "search"
                move, _, _, lines = engine.go(30, allowance_ms=2000)
                report["deadline_fallbacks"][context["bot"]] += "info string deadline-fallback" in lines
                assert move in state["moves"], move
                state = ref.play(move)
                moves.append(move)
                report["plies"] += 1
            replay = ref.export()
            assert ref.call("validate", replay=replay)["valid"]
            (root / "soak replay.json").write_text(json.dumps(replay))
            report["replay_audits"] += 1
            if report["batches"] % 20 == 0:
                for name, extra in (("normal", []), ("proof", ["--replay", str(root / "benchmarks/dynamic-fort-example.json"), "--referee-panel"]), ("reload", ["--replay", str(root / "soak replay.json")])):
                    subprocess.run([str(root / "skallagrim-gui.exe"), "--render", str(root / (name + ".png")), *extra], cwd=root, check=True, timeout=30, capture_output=True)
                    report["render_checks"] += 1
                report["memory_samples"].append(dict(seconds=round(time.monotonic()-started), engine_bytes=[memory(e.proc) for e in engines], referee_bytes=memory(ref.proc)))
            report["batches"] += 1
            save()
            time.sleep(1)
        report["status"] = "pass"
    except BaseException as exc:
        report["status"] = "fail"
        report["failures"].append(repr(exc))
        report["failure_context"] = context
        raise
    finally:
        for engine in engines:
            engine.close()
        if ref is not None:
            ref.close()
        report["owned_processes_exited"] = all(e.proc.poll() is not None for e in engines) and (ref is None or ref.proc.poll() is not None)
        save()


if __name__ == "__main__":
    main()
