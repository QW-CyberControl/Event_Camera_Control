# event_transmission.py
"""Transmission-channel model for event streams.

The channel can apply per-frame bandwidth limits, random packet loss,
timestamp jitter, and fixed latency. It keeps diagnostics so experiments can
report how many events were removed before state estimation.
"""
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from event_buffer import EventBuffer


class EventTransmission:
    """Simulate the path from a DVS sensor to the control processor."""

    def __init__(self, width, height, config=None):
        self.width = width
        self.height = height
        self.config = {
            "bandwidth": -1,      # max events per frame; -1 means unlimited
            "latency_us": 0,      # fixed timestamp offset
            "jitter_us": 0,       # uniform timestamp jitter in [-jitter, +jitter]
            "packet_loss": 0.0,   # independent event drop probability
            "enabled": True,
        }
        if config:
            self.config.update(config)

        self.total_input = 0
        self.total_sent = 0
        self.total_dropped_bandwidth = 0
        self.total_dropped_loss = 0

        print("EventTransmission initialized")
        if self.config["enabled"]:
            print(
                f"  bandwidth={self.config['bandwidth']} ev/frame "
                f"latency={self.config['latency_us']} us "
                f"jitter={self.config['jitter_us']} us "
                f"loss={self.config['packet_loss']}"
            )

    def transmit(self, events):
        """Return events after channel effects have been applied."""
        if events is None or events.i == 0:
            return EventBuffer(1)

        self.total_input += events.i
        if not self.config["enabled"]:
            self.total_sent += events.i
            return events

        x = events.get_x()
        y = events.get_y()
        ts = events.get_ts()
        p = events.get_p()
        n = events.i

        loss = float(self.config["packet_loss"])
        if loss > 0.0:
            keep = np.random.random(n) >= loss
            x, y, ts, p = x[keep], y[keep], ts[keep], p[keep]
            self.total_dropped_loss += n - len(x)
            n = len(x)

        bandwidth = int(self.config["bandwidth"])
        if bandwidth > 0 and n > bandwidth:
            keep_idx = np.random.choice(n, bandwidth, replace=False)
            keep_idx.sort()
            x, y, ts, p = x[keep_idx], y[keep_idx], ts[keep_idx], p[keep_idx]
            self.total_dropped_bandwidth += n - bandwidth
            n = bandwidth

        jitter = int(self.config["jitter_us"])
        if jitter > 0 and n > 0:
            jitter_values = np.random.randint(-jitter, jitter + 1, size=n)
            ts = np.clip(ts.astype(np.int64) + jitter_values, 0, None).astype(np.uint64)

        latency = int(self.config["latency_us"])
        if latency > 0 and n > 0:
            ts = ts.astype(np.uint64) + latency

        out = EventBuffer(max(n, 1))
        if n > 0:
            out.add_array(ts, y, x, p)
        self.total_sent += n
        return out

    def get_statistics(self):
        return {
            "total_input": self.total_input,
            "total_sent": self.total_sent,
            "dropped_bandwidth": self.total_dropped_bandwidth,
            "dropped_loss": self.total_dropped_loss,
            "delivery_ratio": self.total_sent / max(self.total_input, 1),
        }

    def reset(self):
        self.total_input = 0
        self.total_sent = 0
        self.total_dropped_bandwidth = 0
        self.total_dropped_loss = 0
