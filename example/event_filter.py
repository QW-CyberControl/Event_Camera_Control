# event_filter.py
"""Event filtering for the cart-pendulum control pipeline."""
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from event_buffer import EventBuffer


class EventFilter:
    """Filter transmitted events before state estimation or direct control."""

    def __init__(self, width, height, config=None):
        self.width = width
        self.height = height
        self.ground_y = int(height * 2 / 3)

        self.config = {
            "enabled": True,
            "roi_mode": "circle",       # circle or rectangle
            "x_min": 10,
            "x_max": None,
            "y_min": 5,
            "y_max": None,
            "pendulum_length_pixels": None,
            "circle_radius_scale": 1.3,
            "center_y": None,
            "pivot_y_offset": 10,
            "polarity": -1,             # -1 all, 0 ON only, 1 OFF only
            "refractory_us": 500,
            "min_events_per_frame": 0,
            "noise_filter_enabled": False,
            "noise_radius": 3,
            "noise_min_neighbors": 2,
        }
        if config:
            self.config.update(config)

        if self.config["x_max"] is None:
            self.config["x_max"] = width - 10
        if self.config["y_max"] is None:
            self.config["y_max"] = self.ground_y - 25

        pivot_y = self.ground_y - self.config["pivot_y_offset"]
        self.center_y_pixel = self.config["center_y"] or pivot_y
        if self.config["pendulum_length_pixels"] is None:
            self.radius_pixels = None
        else:
            self.radius_pixels = int(
                self.config["pendulum_length_pixels"] *
                self.config["circle_radius_scale"]
            )
        self.cart_x_pixel = None

        self.last_event_time = np.zeros((height, width), dtype=np.uint64)
        self._reset_stats()

        print("EventFilter initialized")
        if self.config["enabled"]:
            if self.config["roi_mode"] == "circle":
                roi = f"circle(center=cart, radius={self.radius_pixels}px)"
            else:
                roi = (
                    f"rectangle x=[{self.config['x_min']}, {self.config['x_max']}] "
                    f"y=[{self.config['y_min']}, {self.config['y_max']}]"
                )
            print(f"  ROI: {roi}")
            print(f"  refractory={self.config['refractory_us']} us")

    def set_cart_x(self, cart_x_pixel):
        self.cart_x_pixel = cart_x_pixel

    def filter(self, events, current_time_us=None):
        """Apply configured filters and return a new EventBuffer."""
        if events is None or events.i == 0:
            return EventBuffer(1)

        self.total_input += events.i
        if not self.config["enabled"]:
            self.total_output += events.i
            return events

        x = events.get_x()
        y = events.get_y()
        ts = events.get_ts()
        p = events.get_p()
        n_before = len(x)

        mask = self._roi_mask(x, y)
        x, y, ts, p = x[mask], y[mask], ts[mask], p[mask]
        self.total_dropped_roi += n_before - len(x)
        if len(x) == 0:
            return EventBuffer(1)

        n_before = len(x)
        polarity = int(self.config["polarity"])
        if polarity >= 0:
            keep_polarity = (p == 1) if polarity == 0 else (p == 0)
            x, y, ts, p = x[keep_polarity], y[keep_polarity], ts[keep_polarity], p[keep_polarity]
            self.total_dropped_polarity += n_before - len(x)
            if len(x) == 0:
                return EventBuffer(1)

        if len(x) < int(self.config["min_events_per_frame"]):
            self.total_dropped_min_events += len(x)
            return EventBuffer(1)

        x, y, ts, p = self._apply_refractory(x, y, ts, p)
        if len(x) == 0:
            return EventBuffer(1)

        if self.config["noise_filter_enabled"] and len(x) > 1:
            x, y, ts, p = self._apply_noise_filter(x, y, ts, p)
            if len(x) == 0:
                return EventBuffer(1)

        out = EventBuffer(max(len(x), 1))
        out.add_array(ts, y, x, p)
        self.total_output += len(x)
        return out

    def _roi_mask(self, x, y):
        cfg = self.config
        if (
            cfg["roi_mode"] == "circle" and
            self.cart_x_pixel is not None and
            self.radius_pixels is not None
        ):
            dx = x.astype(float) - float(self.cart_x_pixel)
            dy = y.astype(float) - float(self.center_y_pixel)
            return (dx * dx + dy * dy) <= self.radius_pixels * self.radius_pixels

        return (
            (x >= cfg["x_min"]) &
            (x < cfg["x_max"]) &
            (y >= cfg["y_min"]) &
            (y < cfg["y_max"])
        )

    def _apply_refractory(self, x, y, ts, p):
        refractory_us = int(self.config["refractory_us"])
        if refractory_us <= 0:
            return x, y, ts, p

        valid = np.ones(len(x), dtype=bool)
        for idx in range(len(x)):
            yi = int(y[idx])
            xi = int(x[idx])
            last_t = self.last_event_time[yi, xi]
            if last_t > 0 and ts[idx] >= last_t and (ts[idx] - last_t) < refractory_us:
                valid[idx] = False
            else:
                self.last_event_time[yi, xi] = ts[idx]

        self.total_dropped_refractory += len(x) - int(np.sum(valid))
        return x[valid], y[valid], ts[valid], p[valid]

    def _apply_noise_filter(self, x, y, ts, p):
        radius = int(self.config["noise_radius"])
        min_neighbors = int(self.config["noise_min_neighbors"])
        keep = np.ones(len(x), dtype=bool)
        xi = x.astype(int)
        yi = y.astype(int)

        for idx in range(len(x)):
            dist = np.abs(xi - xi[idx]) + np.abs(yi - yi[idx])
            neighbors = np.sum((dist > 0) & (dist <= radius))
            if neighbors < min_neighbors:
                keep[idx] = False

        self.total_dropped_noise += len(x) - int(np.sum(keep))
        return x[keep], y[keep], ts[keep], p[keep]

    def get_statistics(self):
        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "dropped_roi": self.total_dropped_roi,
            "dropped_polarity": self.total_dropped_polarity,
            "dropped_noise": self.total_dropped_noise,
            "dropped_refractory": self.total_dropped_refractory,
            "dropped_min_events": self.total_dropped_min_events,
            "filter_ratio": self.total_output / max(self.total_input, 1),
        }

    def reset(self):
        self.last_event_time.fill(0)
        self._reset_stats()

    def _reset_stats(self):
        self.total_input = 0
        self.total_output = 0
        self.total_dropped_roi = 0
        self.total_dropped_polarity = 0
        self.total_dropped_noise = 0
        self.total_dropped_refractory = 0
        self.total_dropped_min_events = 0
