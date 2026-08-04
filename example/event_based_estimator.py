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
            "cart_y_min": self.pivot_y - 12,
            "cart_y_max": self.ground_y + 3,
            "min_events": 10,
            "min_cart_events": 8,
            "window_frames": 3,
            "recent_time_fraction": 0.45,
            "angle_alpha": 0.55,
            "cart_alpha": 0.45,
            "velocity_alpha": 0.25,
            "cart_velocity_alpha": 0.20,
            "velocity_window": 6,
            "max_angle_jump": 0.45,
            "max_cart_jump_m": 0.25,
            "confidence_min": 0.12,
            "ransac_residual_px": 7.0,
            "use_pca_angle": True,
            "initial_angle": 0.0,
            "use_initial_angle_hint": False,
        }
        if config:
            self.config.update(config)

        self.samples = deque(maxlen=self.config["window_frames"])
        self.state_history = deque(maxlen=self.config["velocity_window"])
        self.angle = self._wrap_angle(self.config["initial_angle"])
        self.angle_velocity = 0.0
        self.cart_pos = 0.0
        self.cart_velocity = 0.0
        self.prev_time_us = 0
        self.initialized = bool(self.config["use_initial_angle_hint"])
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
        ts_all = np.concatenate([sample[3] for sample in self.samples])

        if len(x_all) < self.config["min_events"]:
            return self._invalid_result(current_time_us)

        x_all, y_all, p_all = self._keep_recent_events(x_all, y_all, p_all, ts_all)
        if len(x_all) < self.config["min_events"]:
            return self._invalid_result(current_time_us)

        rod_mask = (
            (y_all >= self.config["y_min"]) &
            (y_all <= self.config["y_max"])
        )
        cart_mask = (
            (y_all >= self.config["cart_y_min"]) &
            (y_all <= self.config["cart_y_max"])
        )
        if np.sum(rod_mask) < self.config["min_events"]:
            return self._invalid_result(current_time_us)

        raw = self._fit_state(
            x_all[rod_mask],
            y_all[rod_mask],
            p_all[rod_mask],
            x_all[cart_mask],
        )
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

        self.state_history.append((current_time_us * 1e-6, self.angle, self.cart_pos))
        raw_angle_velocity, raw_cart_velocity = self._fit_velocities()
        if raw_angle_velocity is not None:
            self.angle_velocity = self._ema(
                self.angle_velocity,
                raw_angle_velocity,
                self.config["velocity_alpha"],
            )
        if raw_cart_velocity is not None:
            self.cart_velocity = self._ema(
                self.cart_velocity,
                raw_cart_velocity,
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
        ts = events.get_ts()
        cfg = self.config
        mask = (
            (x >= cfg["x_margin"]) &
            (x < self.width - cfg["x_margin"]) &
            (y >= min(cfg["y_min"], cfg["cart_y_min"])) &
            (y <= max(cfg["y_max"], cfg["cart_y_max"]))
        )
        if np.sum(mask) < 3:
            return None
        return (
            x[mask].astype(float),
            y[mask].astype(float),
            p[mask].astype(float),
            ts[mask].astype(float),
        )

    def _keep_recent_events(self, x, y, p, ts):
        if len(ts) == 0:
            return x, y, p
        fraction = float(self.config["recent_time_fraction"])
        fraction = min(max(fraction, 0.05), 1.0)
        cutoff = np.quantile(ts, 1.0 - fraction)
        keep = ts >= cutoff
        return x[keep], y[keep], p[keep]

    def _fit_state(self, x, y, p, cart_x_events=None):
        y_span = max(float(self.config["y_max"] - self.config["y_min"]), 1.0)
        top_weight = 1.0 + ((self.pivot_y - y) / y_span) ** 2
        polarity_weight = np.where(p > 0, 1.2, 1.0)
        weights = top_weight * polarity_weight

        try:
            slope, intercept = np.polyfit(y, x, 1, w=weights)
            for _ in range(2):
                residual = x - (slope * y + intercept)
                keep = np.abs(residual) <= self.config["ransac_residual_px"]
                if np.sum(keep) < self.config["min_events"]:
                    break
                slope, intercept = np.polyfit(y[keep], x[keep], 1, w=weights[keep])
        except (ValueError, np.linalg.LinAlgError):
            return None

        x_pred = slope * y + intercept
        residual = x - x_pred
        rmse = float(np.sqrt(np.average(residual * residual, weights=weights)))
        spread = float(np.std(x))
        confidence = np.clip((spread / (spread + rmse + 1e-6)) *
                             min(1.0, len(x) / 80.0), 0.0, 1.0)

        angle = float(np.arctan(-slope))
        if self.config["use_pca_angle"]:
            angle = self._pca_angle(x, y, weights, fallback=angle)
        cart_x_pixel = slope * self.pivot_y + intercept
        if cart_x_events is not None and len(cart_x_events) >= self.config["min_cart_events"]:
            # The cart is a rectangle, so the median of its edge events is a
            # stable center estimate when both left/right edges are visible.
            cart_x_pixel = float(np.median(cart_x_events))
        cart_pos = float((cart_x_pixel - self.center_x) / self.pixel_scale)
        return {"angle": angle, "cart_pos": cart_pos, "confidence": confidence}

    def _pca_angle(self, x, y, weights, fallback):
        if len(x) < self.config["min_events"]:
            return fallback
        weight_sum = np.sum(weights)
        if weight_sum <= 0:
            return fallback

        mx = np.sum(weights * x) / weight_sum
        my = np.sum(weights * y) / weight_sum
        coords = np.column_stack((x - mx, y - my))
        cov = (coords * weights[:, None]).T @ coords / weight_sum
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return fallback

        direction = eigvecs[:, int(np.argmax(eigvals))]
        dx, dy = float(direction[0]), float(direction[1])
        if dy > 0:
            dx, dy = -dx, -dy
        angle = np.arctan2(dx, -dy)

        # The rod is orientationless in an event packet, so keep the branch
        # closest to the previous estimate once tracking has started.
        candidates = np.array([angle, angle + np.pi, angle - np.pi])
        reference = self.angle if self.initialized else fallback
        return float(candidates[np.argmin(np.abs(candidates - reference))])

    def _invalid_result(self, current_time_us):
        dt = self._dt_seconds(current_time_us)
        if dt is not None:
            self.angle_velocity *= 0.9
            self.cart_velocity *= 0.9
        self.prev_time_us = current_time_us
        self.confidence = 0.0
        return self.angle, self.angle_velocity, False, self.cart_pos, self.cart_velocity

    def _fit_velocities(self):
        if len(self.state_history) < 3:
            return None, None

        history = np.array(self.state_history, dtype=float)
        t = history[:, 0]
        if t[-1] <= t[0]:
            return None, None

        t0 = t - t.mean()
        angles = np.unwrap(history[:, 1])
        carts = history[:, 2]
        try:
            angle_velocity = np.polyfit(t0, angles, 1)[0]
            cart_velocity = np.polyfit(t0, carts, 1)[0]
        except (ValueError, np.linalg.LinAlgError):
            return None, None
        return float(angle_velocity), float(cart_velocity)

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
        self.state_history.clear()
        self.angle = self._wrap_angle(self.config["initial_angle"])
        self.angle_velocity = 0.0
        self.cart_pos = 0.0
        self.cart_velocity = 0.0
        self.prev_time_us = 0
        self.initialized = bool(self.config["use_initial_angle_hint"])
        self.confidence = 0.0

    @staticmethod
    def _wrap_angle(angle):
        return float((angle + np.pi) % (2 * np.pi) - np.pi)
