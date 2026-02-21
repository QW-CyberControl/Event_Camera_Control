# simple_state_estimator.py
"""
Simplified state estimator for closed-loop control
"""
import numpy as np
import cv2
from collections import deque


class SimpleStateEstimator:
    """
    Simplified state estimator

    Note: For closed-loop testing we can use an inaccurate but fast estimate
    """

    def __init__(self, width, height, config=None):
        self.width = width
        self.height = height

        # Configuration
        self.config = {
            'accumulation_time_us': 20000,
            'history_size': 10,
            'angle_noise_std': 0.1,  # Angle noise standard deviation (rad)
            'velocity_noise_std': 0.5,  # Angular velocity noise standard deviation (rad/s)
            'delay_frames': 2,  # Simulated delay
            'use_ground_truth': False,  # If True, use ground truth (debug)
        }

        if config:
            self.config.update(config)

        # State history
        self.angle_history = deque(maxlen=self.config['history_size'])
        self.velocity_history = deque(maxlen=self.config['history_size'])
        self.time_history = deque(maxlen=self.config['history_size'])

        # Delay buffer
        self.delay_buffer = deque(maxlen=self.config['delay_frames'])

        # Event accumulation image
        self.event_accumulator = np.zeros((height, width), dtype=np.float32)

        # State
        self.current_angle = 0.0
        self.current_velocity = 0.0

        print("Simplified state estimator initialized")

    def estimate_from_events(self, events, current_time_us):
        """
        Estimate state from event stream

        Args:
            events: event buffer
            current_time_us: current time (microseconds)

        Returns:
            angle, angular_velocity, valid
        """
        # Simplified estimation: assume we know the true state, add noise and delay
        # In a real system, the state should be extracted from the event stream

        # Method 1: if ground truth is allowed (debug)
        if self.config['use_ground_truth'] and hasattr(self, 'ground_truth_callback'):
            true_angle, true_velocity = self.ground_truth_callback()

            # Add noise
            angle = true_angle + np.random.normal(0, self.config['angle_noise_std'])
            velocity = true_velocity + np.random.normal(0, self.config['velocity_noise_std'])

            valid = True

        else:
            # Method 2: simple event processing
            # This is just an example; a proper event‑to‑angle conversion needs to be implemented
            if events.i > 0:
                # Simple event counting method
                event_density = events.i / (self.width * self.height)

                # Assume event density is related to the rate of change of angle
                angle_change = event_density * 0.1  # scaling factor

                # Update angle
                self.current_angle += angle_change

                # Constrain angle range
                if self.current_angle > np.pi:
                    self.current_angle -= 2 * np.pi
                elif self.current_angle < -np.pi:
                    self.current_angle += 2 * np.pi

                # Compute angular velocity (simplified)
                if len(self.angle_history) > 0:
                    dt = current_time_us * 1e-6 - self.time_history[-1]
                    if dt > 0:
                        self.current_velocity = (self.current_angle - self.angle_history[-1]) / dt
                    else:
                        self.current_velocity = 0.0
                else:
                    self.current_velocity = 0.0

                valid = True
            else:
                # No events, keep current estimate
                valid = False

        # Add to history
        self.angle_history.append(self.current_angle)
        self.velocity_history.append(self.current_velocity)
        self.time_history.append(current_time_us * 1e-6)

        # Add delay
        self.delay_buffer.append((self.current_angle, self.current_velocity))

        # Use delayed state
        if len(self.delay_buffer) >= self.config['delay_frames']:
            delayed_angle, delayed_velocity = self.delay_buffer[0]
        else:
            delayed_angle, delayed_velocity = self.current_angle, self.current_velocity

        return delayed_angle, delayed_velocity, valid

    def set_ground_truth_callback(self, callback):
        """Set ground truth callback function (for debugging)"""
        self.ground_truth_callback = callback

    def reset(self):
        """Reset the estimator"""
        self.angle_history.clear()
        self.velocity_history.clear()
        self.time_history.clear()
        self.delay_buffer.clear()
        self.event_accumulator.fill(0)
        self.current_angle = 0.0
        self.current_velocity = 0.0

        print("State estimator reset")
