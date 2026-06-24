"""
load_generator.py — UI-controllable traffic generator for the live demo.

Lets the web console start/stop a real traffic wave against the cluster app
(so you don't need a separate terminal). It only sends requests — it never
scales anything; the console's agent does the scaling.

It port-forwards the service to localhost:8090 and runs worker threads whose
intensity follows a triangular wave (ramp up, then down) over the chosen
duration, then auto-stops.
"""

import time
import random
import threading
import subprocess
import urllib.request
from collections import deque

DEPLOYMENT = "eco-sample-app"
URL = "http://localhost:8090/work"
N_WORKERS = 60


class LoadGenerator:
    """Start/stop a triangular traffic wave; report live intensity + latency."""

    def __init__(self):
        self._running = False
        self._intensity = 0.0
        self._pf = None
        self._threads = []
        self._recent = deque(maxlen=400)
        self._start = None
        self._duration = 0

    # ── worker + wave ────────────────────────────────────────────
    def _worker(self):
        while self._running:
            if random.random() < self._intensity:
                t0 = time.perf_counter()
                try:
                    urllib.request.urlopen(URL, timeout=5).read()
                    self._recent.append(time.perf_counter() - t0)
                except Exception:
                    self._recent.append(5.0)
            else:
                time.sleep(0.05)

    def _driver(self):
        """Update intensity along the wave; auto-stop when the duration elapses."""
        while self._running:
            elapsed = time.perf_counter() - self._start
            if elapsed >= self._duration:
                self.stop()
                break
            progress = elapsed / self._duration
            self._intensity = 1.0 - abs(2 * progress - 1)     # peak at the midpoint
            time.sleep(0.5)

    # ── control ──────────────────────────────────────────────────
    def start(self, duration=300):
        if self._running:
            return self.status()
        self._pf = subprocess.Popen(
            ["kubectl", "port-forward", f"svc/{DEPLOYMENT}", "8090:80"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(4)                       # let the port-forward come up
        self._running = True
        self._duration = max(10, int(duration))
        self._start = time.perf_counter()
        self._recent.clear()
        self._threads = [threading.Thread(target=self._worker, daemon=True)
                         for _ in range(N_WORKERS)]
        for t in self._threads:
            t.start()
        threading.Thread(target=self._driver, daemon=True).start()
        return self.status()

    def stop(self):
        self._running = False
        self._intensity = 0.0
        if self._pf:
            self._pf.terminate()
            self._pf = None
        return self.status()

    def status(self):
        lat = sorted(self._recent)
        p95 = lat[int(0.95 * (len(lat) - 1))] * 1000 if lat else 0.0
        elapsed = (time.perf_counter() - self._start) if (self._running and self._start) else 0.0
        return {
            "running": self._running,
            "intensity": round(self._intensity, 2),
            "elapsed": round(elapsed, 1),
            "duration": self._duration,
            "p95_ms": round(p95, 0),
        }
