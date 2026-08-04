# pendulum_controller.py
"""Controllers for the inverted pendulum on a cart."""
import numpy as np


class PendulumController:
    """PD, PID, LQR, BangBang, and hybrid swing-up/LQR controller."""

    def __init__(self, config=None):
        self.config = {
            "controller_type": "LQR",
            "K_cart_pos": -3.1623,
            "K_cart_vel": -5.0317,
            "K_angle": 47.7304,
            "K_angle_vel": 13.979,
            "Kp": 50.0,
            "Kd": 10.0,
            "Ki": 0.0,
            "integral_limit": 5.0,
            "max_force": 12.0,
            "target_angle": 0.0,
            "target_cart": 0.0,
            "deadband": 0.005,
            "sampling_rate": 100.0,
            "pendulum_mass": 0.1,
            "pendulum_length": 1.0,
            "gravity": 9.81,
            "swingup_gain": 10.0,
            "swingup_cart_gain": 4.0,
            "swingup_cart_vel_gain": 4.0,
            "swingup_kick_force": 8.0,
            "lqr_switch_angle": np.radians(22.0),
            "lqr_switch_velocity": 4.5,
        }
        if config:
            self.config.update(config)

        self.integral_error = 0.0
        self.prev_error = 0.0
        self.control_history = []
        self.error_history = []
        self.cart_error_history = []
        self.mode_history = []
        self.current_mode = self.config["controller_type"]

        print(f"Controller initialized: {self.config['controller_type']} controller")
        print(
            f"  LQR gains: Kx={self.config['K_cart_pos']}, "
            f"Kv={self.config['K_cart_vel']}, Kth={self.config['K_angle']}, "
            f"Kw={self.config['K_angle_vel']}"
        )

    def compute_control(self, angle, angular_velocity, current_time=None,
                        cart_pos=0.0, cart_velocity=0.0):
        """Compute the cart force from the estimated state."""
        controller_type = self.config["controller_type"]
        angle = self._wrap_angle(angle)
        angle_error = self._wrap_angle(self.config["target_angle"] - angle)
        if abs(angle_error) < self.config["deadband"]:
            angle_error = 0.0

        self.integral_error += angle_error / self.config["sampling_rate"]
        self.integral_error = np.clip(
            self.integral_error,
            -self.config["integral_limit"],
            self.config["integral_limit"],
        )

        if controller_type == "SwingUpLQR":
            control_force = self._compute_swingup_lqr(
                angle,
                angular_velocity,
                cart_pos,
                cart_velocity,
            )
        elif controller_type == "LQR":
            self.current_mode = "LQR"
            control_force = self._compute_lqr(angle, angular_velocity, cart_pos, cart_velocity)
        elif controller_type == "PD":
            self.current_mode = "PD"
            control_force = self.config["Kp"] * angle_error - self.config["Kd"] * angular_velocity
        elif controller_type == "PID":
            self.current_mode = "PID"
            control_force = (
                self.config["Kp"] * angle_error
                - self.config["Kd"] * angular_velocity
                + self.config["Ki"] * self.integral_error
            )
        elif controller_type == "BangBang":
            self.current_mode = "BangBang"
            control_force = self.config["max_force"] if angle_error > 0 else -self.config["max_force"]
        else:
            self.current_mode = "PD"
            control_force = self.config["Kp"] * angle_error - self.config["Kd"] * angular_velocity

        control_force = float(np.clip(
            control_force,
            -self.config["max_force"],
            self.config["max_force"],
        ))

        self.control_history.append(control_force)
        self.error_history.append(angle_error)
        self.cart_error_history.append(self.config["target_cart"] - cart_pos)
        self.mode_history.append(self.current_mode)
        return control_force

    def _compute_swingup_lqr(self, angle, angular_velocity, cart_pos, cart_velocity):
        near_upright = (
            abs(angle) <= self.config["lqr_switch_angle"] and
            abs(angular_velocity) <= self.config["lqr_switch_velocity"]
        )
        if near_upright:
            self.current_mode = "LQR"
            return self._compute_lqr(angle, angular_velocity, cart_pos, cart_velocity)

        self.current_mode = "SwingUp"
        m = self.config["pendulum_mass"]
        length = self.config["pendulum_length"]
        gravity = self.config["gravity"]
        desired_energy = m * gravity * length
        energy = 0.5 * m * (length * angular_velocity) ** 2 + m * gravity * length * np.cos(angle)
        energy_error = energy - desired_energy

        swing_force = (
            -self.config["swingup_gain"] *
            energy_error *
            angular_velocity *
            np.cos(angle)
        )
        if abs(angular_velocity) < 0.05 and abs(abs(angle) - np.pi) < 0.25:
            swing_force = self.config["swingup_kick_force"]
        cart_centering = (
            -self.config["swingup_cart_gain"] * cart_pos
            -self.config["swingup_cart_vel_gain"] * cart_velocity
        )
        return swing_force + cart_centering

    def _compute_lqr(self, angle, angular_velocity, cart_pos, cart_velocity):
        return -(
            self.config["K_cart_pos"] * cart_pos
            + self.config["K_cart_vel"] * cart_velocity
            + self.config["K_angle"] * angle
            + self.config["K_angle_vel"] * angular_velocity
        )

    @staticmethod
    def _wrap_angle(angle):
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def reset(self):
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.control_history = []
        self.error_history = []
        self.cart_error_history = []
        self.mode_history = []
        self.current_mode = self.config["controller_type"]

    def get_control_statistics(self):
        if not self.control_history:
            return {
                "avg_force": 0.0,
                "max_force": 0.0,
                "rms_error": 0.0,
                "rms_cart_error": 0.0,
                "num_control_steps": 0,
                "mode_counts": {},
            }

        controls = np.array(self.control_history)
        errors = np.array(self.error_history)
        cart_errors = np.array(self.cart_error_history)
        mode_counts = {
            mode: self.mode_history.count(mode)
            for mode in sorted(set(self.mode_history))
        }
        return {
            "avg_force": float(np.mean(np.abs(controls))),
            "max_force": float(np.max(np.abs(controls))),
            "rms_error": float(np.sqrt(np.mean(errors ** 2))),
            "rms_cart_error": float(np.sqrt(np.mean(cart_errors ** 2))),
            "num_control_steps": len(self.control_history),
            "mode_counts": mode_counts,
        }
