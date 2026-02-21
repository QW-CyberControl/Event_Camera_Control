# pendulum_controller.py
"""
Inverted pendulum controller module
"""
import numpy as np


class PendulumController:
    """Inverted pendulum controller"""

    def __init__(self, config=None):
        self.config = {
            'controller_type': 'PD',  # Controller type: PD, PID, LQR, BangBang
            'Kp': 50.0,  # Proportional gain
            'Kd': 10.0,  # Derivative gain
            'Ki': 0.0,  # Integral gain
            'max_force': 10.0,  # Maximum control force (N)
            'target_angle': 0.0,  # Target angle (rad) – upright
            'integral_limit': 5.0,  # Integral term limit
            'deadband': 0.01,  # Deadband (rad)
            'sampling_rate': 100.0,  # Sampling rate (Hz)
        }

        if config:
            self.config.update(config)

        # Controller state
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_time = 0.0

        # Control history
        self.control_history = []
        self.error_history = []

        print(f"Controller initialized: {self.config['controller_type']} controller")
        print(f"  Gains: Kp={self.config['Kp']}, Kd={self.config['Kd']}, Ki={self.config['Ki']}")

    def compute_control(self, angle, angular_velocity, current_time=None):
        """
        Compute control force

        Args:
            angle: current estimated angle (rad)
            angular_velocity: current estimated angular velocity (rad/s)
            current_time: current time (s), used for derivative calculation

        Returns:
            control force (N)
        """
        # Compute angle error (target angle is 0, upright)
        error = self.config['target_angle'] - angle

        # Deadband handling
        if abs(error) < self.config['deadband']:
            error = 0.0

        # Update integral term
        self.integral_error += error / self.config['sampling_rate']

        # Integral term limit
        self.integral_error = np.clip(
            self.integral_error,
            -self.config['integral_limit'],
            self.config['integral_limit']
        )

        # Compute control force based on controller type
        if self.config['controller_type'] == 'PD':
            # PD controller
            control_force = (
                    self.config['Kp'] * error +
                    self.config['Kd'] * (-angular_velocity)  # 负号因为要阻尼
            )

        elif self.config['controller_type'] == 'PID':
            # PID controller
            control_force = (
                    self.config['Kp'] * error +
                    self.config['Kd'] * (-angular_velocity) +
                    self.config['Ki'] * self.integral_error
            )

        elif self.config['controller_type'] == 'LQR':
            # Simplified LQR controller
            # Here approximated by PD
            control_force = (
                    self.config['Kp'] * error +
                    self.config['Kd'] * (-angular_velocity)
            )

        elif self.config['controller_type'] == 'BangBang':
            # Bang‑Bang controller
            if error > 0:
                control_force = self.config['max_force']
            else:
                control_force = -self.config['max_force']

        else:
            # Default PD controller
            control_force = (
                    self.config['Kp'] * error +
                    self.config['Kd'] * (-angular_velocity)
            )

        # Limit control force
        control_force = np.clip(
            control_force,
            -self.config['max_force'],
            self.config['max_force']
        )

        # Record history
        self.control_history.append(control_force)
        self.error_history.append(error)

        return control_force

    def reset(self):
        """Reset the controller"""
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.control_history = []
        self.error_history = []

        print("Controller reset")

    def get_control_statistics(self):
        """Get control statistics"""
        if not self.control_history:
            return {
                'avg_force': 0.0,
                'max_force': 0.0,
                'rms_error': 0.0
            }

        avg_force = np.mean(np.abs(self.control_history))
        max_force = np.max(np.abs(self.control_history))

        if self.error_history:
            rms_error = np.sqrt(np.mean(np.array(self.error_history) ** 2))
        else:
            rms_error = 0.0

        return {
            'avg_force': avg_force,
            'max_force': max_force,
            'rms_error': rms_error,
            'num_control_steps': len(self.control_history)
        }