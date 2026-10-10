"""Line protocol process transport; no shell, bounded reads, explicit teardown."""

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_NAME = "skallagrim.exe" if os.name == "nt" else "skallagrim"
ENGINE = Path(
    os.environ.get(
        "SKALLAGRIM_ENGINE",
        str(
            ROOT / ENGINE_NAME
            if (ROOT / ENGINE_NAME).is_file()
            else ROOT / "target/release" / ENGINE_NAME
        ),
    )
)


def pin_process(process, cpu):
    if not 0 <= cpu < (os.cpu_count() or 1):
        raise ValueError("logical CPU out of range")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = kernel.SetProcessAffinityMask
        fn.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        fn.restype = wintypes.BOOL
        if not fn(process._handle, 1 << cpu):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.sched_setaffinity(process.pid, {cpu})


class Engine:
    def __init__(self, command=None, options=None, cwd=None, stderr=None, cpu=None):
        self.lines = queue.Queue()
        self.proc = subprocess.Popen(
            command or [str(ENGINE)],
            cwd=cwd or ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr or subprocess.DEVNULL,
            text=True,
            bufsize=1,
            start_new_session=os.name != "nt",
            env=os.environ | ({"SKALLAGRIM_CPU": str(cpu)} if cpu is not None else {}),
        )

        def pump():
            try:
                for line in self.proc.stdout:
                    self.lines.put(line.rstrip())
            finally:
                self.lines.put(None)

        self.reader = threading.Thread(target=pump, daemon=True)
        self.reader.start()
        try:
            if cpu is not None:
                pin_process(self.proc, cpu)
            self.send("uci")
            self.until("uciok", 120)
            for name, value in (options or {}).items():
                self.send(f"setoption name {name} value {value}")
            self.ready()
        except BaseException:
            self.close()
            raise

    def send(self, command):
        if "\n" in command or "\r" in command:
            raise ValueError("protocol command contains newline")
        if self.proc.poll() is not None:
            raise RuntimeError(f"engine exited with {self.proc.returncode}")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def until(self, prefix, timeout=5):
        deadline = time.perf_counter() + timeout
        seen = []
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"waiting for {prefix}")
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"waiting for {prefix}") from None
            if line is None:
                raise RuntimeError("engine closed stdout")
            seen.append(line)
            if line.startswith("info string error"):
                raise RuntimeError(line)
            if line == prefix or line.startswith(prefix + " "):
                return line[len(prefix) :].strip(), seen

    def ready(self):
        self.send("isready")
        self.until("readyok", 30)

    def position(self, moves, fen=None):
        base = f"fen {fen}" if fen else "startpos"
        self.send("position " + base + (" moves " + " ".join(moves) if moves else ""))
        self.ready()

    def snapshot(self):
        self.send("snapshot")
        return json.loads(self.until("snapshot")[0])

    def play(self, move):
        self.send("play " + move)
        self.send("snapshot")
        return json.loads(self.until("snapshot")[0])

    def evaluate(self):
        self.send("eval")
        return json.loads(self.until("eval")[0])

    def go(self, milliseconds, allowance_ms=1000, nodes=None):
        started = time.perf_counter()
        self.send(f"go movetime {milliseconds}" + (f" nodes {nodes}" if nodes else ""))
        move, lines = self.until("bestmove", (milliseconds + allowance_ms) / 1000)
        elapsed = (time.perf_counter() - started) * 1000
        policy = []
        for line in lines:
            if line.startswith("info string policy "):
                policy = json.loads(line[len("info string policy ") :])
        return move, elapsed, policy, lines

    def close(self):
        if self.proc.poll() is None:
            try:
                self.send("quit")
                self.proc.wait(timeout=2)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=3,
                    )
                else:
                    import signal

                    os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=2)
        self.reader.join(timeout=1)
        for stream in (
            self.proc.stdin,
            self.proc.stdout if not self.reader.is_alive() else None,
        ):
            if stream:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
