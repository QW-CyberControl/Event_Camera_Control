# event_based_estimator.py
"""
Estimate the cart-pendulum state from filtered event-camera output.

The estimator fits the pendulum rod in image space over a short sliding
window. For a rendered pendulum, rod events approximately satisfy:

    x_pixel = cart_x_pixel - tan(theta) * (y_pixel - pivot_y_pixel)

Therefore a weighted line fit gives both pendulum angle and cart position.
Velocities are obtained by differentiating the smoothed estimates.
"""
from collections import deque

import numpy as np


class EventBasedEstimator:
    """Four-state estimator: angle, angular velocity, cart position, cart velocity."""

    def __init__(self, width, height, config=None):
        self.width = width
        self.height = height

        ground_y = int(height * 2 / 3)
        self.ground_y = ground_y
        self.pivot_y = ground_y - 10
        self.center_x = width // 2

        rod_length_m = 1.0
        self.pixel_scale = min(width / (4 * rod_length_m),
                               (ground_y - 20) / (2 * rod_length_m))
        rod_len_px = int(rod_length_m * self.pixel_scale)

        self.config = {
            "x_margin": 12,
            "y_min": max(5, self.pivot_y - rod_len_px - 12),
            "y_max": self.pivot_y + 4,
            "min_events": 10,
            "window_frames": 3,
            "angle_alpha": 0.55,
            "cart_alpha": 0.45,
            "velocity_alpha": 0.25,
            "cart_velocity_alpha": 0.25,
            "max_angle_jump": 0.45,
            "max_cart_jump_m": 0.25,
            "confidence_min": 0.15,
        }
        if config:
            self.config.update(config)

        self.samples = deque(maxlen=self.config["window_frames"])
        self.angle = 0.0
        self.angle_velocity = 0.0
        self.cart_pos = 0.0
        self.cart_velocity = 0.0
        self.prev_time_us = 0
        self.initialized = False
        self.confidence = 0.0

        print("EventBasedEstimator initialized")
        print(f"  window={self.config['window_frames']} frames")
        print(f"  ROI y=[{self.config['y_min']}, {self.config['y_max']}]")

    def estimate_from_events(self, events, current_time_us):
        """Return (angle, angular_velocity, valid, cart_pos, cart_velocity)."""
        filtered = self._filter_events(events)
        if filtered is not None:
            self.samples.append(filtered)

        if len(self.samples) < self.config["window_frames"]:
            return self._invalid_result(current_time_us)

        x_all = np.concatenate([sample[0] for sample in self.samples])
        y_all = np.concatenate([sample[1] for sample in self.samples])
        p_all = np.concatenate([sample[2] for sample in self.samples])

        if len(x_all) < self.config["min_events"]:
            return self._invalid_result(current_time_us)

        raw = self._fit_state(x_all, y_all, p_all)
        if raw is None or raw["confidence"] < self.config["confidence_min"]:
            return self._invalid_result(current_time_us)

        prev_angle = self.angle
        prev_cart = self.cart_pos
        raw_angle = self._limit_jump(raw["angle"], self.angle,
                                     self.config["max_angle_jump"])
        raw_cart = self._limit_jump(raw["cart_pos"], self.cart_pos,
                                    self.config["max_cart_jump_m"])

        if not self.initialized:
            self.angle = raw_angle
            self.cart_pos = raw_cart
            self.initialized = True
        else:
            self.angle = self._ema(self.angle, raw_angle,
                                   self.config["angle_alpha"])
            self.cart_pos = self._ema(self.cart_pos, raw_cart,
                                      self.config["cart_alpha"])

        dt = self._dt_seconds(current_time_us)
        if dt is not None:
            raw_angle_velocity = (self.angle - prev_angle) / dt
            raw_cart_velocity = (self.cart_pos - prev_cart) / dt
            self.angle_velocity = self._ema(
                self.angle_velocity, raw_angle_velocity,
                self.config["velocity_alpha"],
            )
            self.cart_velocity = self._ema(
                self.cart_velocity, raw_cart_velocity,
                self.config["cart_velocity_alpha"],
            )

        self.prev_time_us = current_time_us
        self.confidence = raw["confidence"]
        return self.angle, self.angle_velocity, True, self.cart_pos, self.cart_velocity

    def _filter_events(self, events):
        if events is None or events.i == 0:
            return None

        x = events.get_x()
        y = events.get_y()
        p = events.get_p()
        cfg = self.config
        mask = (
            (x >= cfg["x_margin"]) &
            (x < self.width - cfg["x_margin"]) &
            (y >= cfg["y_min"]) &
            (y <= cfg["y_max"])
        )
        if np.sum(mask) < 3:
            return None
        return x[mask].astype(float), y[mask].astype(float), p[mask].astype(float)

    def _fit_state(self, x, y, p):
        y_span = max(float(self.config["y_max"] - self.config["y_min"]), 1.0)
        top_weight = 1.0 + ((self.pivot_y - y) / y_span) ** 2
        polarity_weight = np.where(p > 0, 1.2, 1.0)
        weights = top_weight * polarity_weight

        try:
            slope, intercept = np.polyfit(y, x, 1, w=weights)
        except (ValueError, np.linalg.LinAlgError):
            return None

        x_pred = slope * y + intercept
        residual = x - x_pred
        rmse = float(np.sqrt(np.average(residual * residual, weights=weights)))
        spread = float(np.std(x))
        confidence = np.clip((spread / (spread + rmse + 1e-6)) *
                             min(1.0, len(x) / 80.0), 0.0, 1.0)

        angle = float(np.arctan(-slope))
        cart_x_pixel = slope * self.pivot_y + intercept
        cart_pos = float((cart_x_pixel - self.center_x) / self.pixel_scale)
        return {"angle": angle, "cart_pos": cart_pos, "confidence": confidence}

    def _invalid_result(self, current_time_us):
        dt = self._dt_seconds(current_time_us)
        if dt is not None:
            self.angle_velocity *= 0.9
            self.cart_velocity *= 0.9
        self.prev_time_us = current_time_us
        self.confidence = 0.0
        return self.angle, self.angle_velocity, False, self.cart_pos, self.cart_velocity

    def _dt_seconds(self, current_time_us):
        if self.prev_time_us <= 0 or current_time_us <= self.prev_time_us:
            return None
        dt = (current_time_us - self.prev_time_us) * 1e-6
        return dt if dt > 1e-9 else None

    @staticmethod
    def _ema(previous, current, alpha):
        return alpha * current + (1.0 - alpha) * previous

    @staticmethod
    def _limit_jump(current, previous, max_jump):
        if max_jump <= 0:
            return current
        delta = current - previous
        if abs(delta) <= max_jump:
            return current
        return previous + np.sign(delta) * max_jump

    def reset(self):
        self.samples.clear()
        self.angle = 0.0
        self.angle_velocity = 0.0
        self.cart_pos = 0.0
        self.cart_velocity = 0.0
        self.prev_time_us = 0
        self.initialized = False
        self.confidence = 0.0
