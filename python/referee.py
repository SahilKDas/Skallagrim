"""Dedicated Rust referee JSON-line client. Board and clocks remain in the child."""

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from transport import ROOT

REFEREE_NAME = "skallagrim-referee.exe" if os.name == "nt" else "skallagrim-referee"


class MatchTimeout(TimeoutError):
    """A synchronized referee reply adjudicating the game clock."""


class Referee:
    def __init__(self, executable=None, log_dir=None):
        self.log_dir = Path(log_dir) if log_dir else ROOT / "runs/referee"
        self.executable = Path(
            executable
            or os.environ.get(
                "SKALLAGRIM_REFEREE",
                ROOT / REFEREE_NAME
                if (ROOT / REFEREE_NAME).is_file()
                else ROOT / "target/release" / REFEREE_NAME,
            )
        )
        self.proc = subprocess.Popen(
            [str(self.executable), "--log-dir", str(log_dir or ROOT / "runs/referee")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf8",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.replies = queue.Queue()
        self.state = {}
        self.sequence = 0
        self.failed = False
        self.latencies = []

        def read():
            try:
                for line in self.proc.stdout:
                    self.replies.put(line)
            finally:
                self.replies.put(None)

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        try:
            self.snapshot()
        except BaseException:
            self.close()
            raise

    def call(self, command, **fields):
        if self.failed:
            raise RuntimeError("referee connection unusable after protocol failure")
        started = time.perf_counter()
        try:
            return self._call(command, **fields)
        except MatchTimeout:
            raise
        except (RuntimeError, OSError, json.JSONDecodeError, KeyError):
            self.failed = True
            raise
        finally:
            self.latencies.append(
                dict(
                    command=command,
                    milliseconds=(time.perf_counter() - started) * 1000,
                    failed=self.failed,
                )
            )
            # Bounded diagnostics, independent of the referee game clock.
            del self.latencies[:-256]
            if self.failed:
                try:
                    self.log_dir.mkdir(parents=True, exist_ok=True)
                    (
                        self.log_dir / f"protocol-failure-{self.proc.pid}.json"
                    ).write_text(
                        json.dumps(
                            dict(
                                sequence=self.sequence,
                                command=command,
                                revision=self.state.get("revision"),
                                response=getattr(self, "protocol_failure", None),
                                latencies=self.latencies,
                            )
                        ),
                        encoding="utf-8",
                    )
                except OSError:
                    pass  # Do not mask the original protocol failure.

    def _call(self, command, **fields):
        self.sequence += 1
        request = dict(command=command, id=self.sequence, **fields)
        if (
            command
            not in {
                "snapshot",
                "new",
                "export",
                "pause",
                "validate",
                "forfeit",
                "limit",
            }
            and "revision" in self.state
        ):
            request.setdefault("revision", self.state["revision"])
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        try:
            line = self.replies.get(timeout=15 if command == "validate" else 5)
        except queue.Empty as exc:
            raise RuntimeError("referee response deadline") from exc
        if line is None:
            raise RuntimeError("referee exited")
        response = json.loads(line)
        if response.get("id") != self.sequence:
            self.protocol_failure = dict(
                expected_id=self.sequence,
                received_id=response.get("id"),
                reply=line[:8192],
            )
            raise RuntimeError("referee response mismatch")
        self.state = response["state"]
        if not response["ok"]:
            if (self.state.get("outcome") or {}).get("reason") == "timeout":
                raise MatchTimeout(response["error"])
            raise ValueError(response["error"])
        return response["result"]

    def snapshot(self):
        return self.call("snapshot")

    def position(self, moves=(), fen=None):
        fields = {"moves": list(moves)}
        if fen is not None:
            fields["fen"] = fen
        return self.call("new", **fields)

    def play(self, move):
        return self.call("play", move=move)

    def start_clock(self, milliseconds):
        return self.call("clock", milliseconds=milliseconds)

    def export(self):
        return self.call("export")

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"command":"quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=0.5)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait(timeout=2)
        for stream in [self.proc.stdin, self.proc.stdout]:
            if stream:
                try:
                    stream.close()
                except OSError:
                    pass
        self.reader.join(timeout=0.5)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
