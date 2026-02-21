# pendulum_simulator.py
"""
Inverted pendulum physics simulator
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class InvertedPendulumSimulator:
    """Inverted pendulum physics simulator"""

    def __init__(self, config=None):
        """Initialize the inverted pendulum simulator"""
        # Physical parameters
        self.config = {
            'length': 1.0,                  # Pendulum length (m)
            'mass': 0.1,                    # Pendulum mass (kg)
            'gravity': 9.81,                # Gravity acceleration (m/s²)
            'friction': 0.1,                # Friction coefficient
            'cart_mass': 1.0,               # Cart mass (kg)
            'cart_friction': 0.05,          # Cart friction coefficient

            # Image parameters
            'image_width': 320,
            'image_height': 240,
            'pendulum_thickness': 5,
            'cart_width': 40,
            'cart_height': 20,

            # Control parameters
            'max_force': 10.0,  # Maximum control force (N)
            'sampling_rate': 100.0,  # Sampling rate (Hz)

            # Initial state
            'initial_angle': np.radians(5),   # Initial angle (rad)
            'initial_angular_velocity': 0.0,  # Initial angular velocity (rad/s)
            'initial_cart_position': 0.0,     # Initial cart position (m)
            'initial_cart_velocity': 0.0,     # Initial cart velocity (m/s)
        }

        if config:
            self.config.update(config)

        # State variables
        self.state = np.array([
            self.config['initial_cart_position'],
            self.config['initial_cart_velocity'],
            self.config['initial_angle'],
            self.config['initial_angular_velocity']
        ])

        self.time = 0.0
        self.dt = 1.0 / self.config['sampling_rate']

        # Control input history
        self.control_history = []
        self.state_history = []
        self.time_history = []

        # Image frame buffer
        self.frame_buffer = []

        # Pre‑computed parameters
        self.m = self.config['mass']
        self.M = self.config['cart_mass']
        self.l = self.config['length']
        self.g = self.config['gravity']
        self.b = self.config['friction']
        self.b_cart = self.config['cart_friction']

        print("Inverted pendulum simulator initialized:")
        print(f"  Pendulum length: {self.l}m, mass: {self.m}kg")
        print(f"  Cart mass: {self.M}kg")
        print(f"  Sampling rate: {self.config['sampling_rate']}Hz")
        print(f"  Initial angle: {np.degrees(self.state[2]):.1f}°")

    def dynamics(self, t, state, F):
        """
        Inverted pendulum dynamics equations

        Args:
            t: time
            state: [x, x_dot, theta, theta_dot]
            F: control force (N)

        Returns:
            state derivatives
        """
        x, x_dot, theta, theta_dot = state

        # System parameters
        m, M, l, g, b, b_cart = self.m, self.M, self.l, self.g, self.b, self.b_cart

        # Intermediate variables
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Compute accelerations
        denominator = (M + m) * (l ** 2) - m ** 2 * l ** 2 * cos_theta ** 2

        if abs(denominator) < 1e-6:
            denominator = 1e-6

        # Angular acceleration
        theta_ddot = (
                             (M + m) * g * sin_theta
                             - m * l * theta_dot ** 2 * sin_theta * cos_theta
                             + (F - b_cart * x_dot) * cos_theta
                             - b * theta_dot
                     ) / denominator * l

        # Cart acceleration
        x_ddot = (
                         F - b_cart * x_dot
                         + m * l * (theta_dot ** 2 * sin_theta - theta_ddot * cos_theta)
                 ) / (M + m)

        return [x_dot, x_ddot, theta_dot, theta_ddot]

    def step(self, control_force=0.0):
        """
        Perform one simulation time step

        Args:
            control_force: control force (N)

        Returns:
            updated state
        """
        # Limit control force
        control_force = np.clip(
            control_force,
            -self.config['max_force'],
            self.config['max_force']
        )

        # RK4 integration
        k1 = self.dynamics(self.time, self.state, control_force)
        k2 = self.dynamics(self.time + self.dt / 2,
                           self.state + np.array(k1) * self.dt / 2,
                           control_force)
        k3 = self.dynamics(self.time + self.dt / 2,
                           self.state + np.array(k2) * self.dt / 2,
                           control_force)
        k4 = self.dynamics(self.time + self.dt,
                           self.state + np.array(k3) * self.dt,
                           control_force)

        # Update state
        self.state += (np.array(k1) + 2 * np.array(k2) + 2 * np.array(k3) + np.array(k4)) * self.dt / 6

        # Update time
        self.time += self.dt

        # Record history
        self.state_history.append(self.state.copy())
        self.control_history.append(control_force)
        self.time_history.append(self.time)

        return self.state.copy()

    def get_current_image(self):
        """
        Generate an image of the current state

        Returns:
            BGR image of the current state
        """
        width = self.config['image_width']
        height = self.config['image_height']

        # Create blank image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Compute pendulum endpoint position
        x_cart = self.state[0]  # cart position
        theta = self.state[2]  # pendulum angle

        # Convert physical coordinates to image coordinates
        # Image centre corresponds to x=0, bottom corresponds to y=0
        ground_y = int(height * 2 / 3)
        vertical_scale = (ground_y - 20) / (2 * self.l)
        # Horizontal: leave 1/4 width on each side, so that pendulum swings within bounds
        horizontal_scale = width / (4 * self.l)
        # Take the smaller value to ensure it fits in both directions
        scale = min(horizontal_scale, vertical_scale)  # scaling factor
        # scale = min(width, height) / (3 * self.l)

        center_x = width // 2


        # Cart position
        cart_x = int(center_x + x_cart * scale)
        cart_y = ground_y

        # Pendulum endpoint
        pendulum_length_pixels = int(self.l * scale)
        pendulum_end_x = int(cart_x + pendulum_length_pixels * np.sin(theta))
        pendulum_end_y = int(cart_y - pendulum_length_pixels * np.cos(theta))

        # Draw ground
        cv2.line(image, (0, ground_y), (width, ground_y), (200, 200, 200), 2)

        # Draw cart
        cart_width = self.config['cart_width']
        cart_height = self.config['cart_height']
        cv2.rectangle(image,
                      (cart_x - cart_width // 2, cart_y - cart_height),
                      (cart_x + cart_width // 2, cart_y),
                      (100, 100, 255), -1)

        # Draw pendulum rod
        pendulum_thickness = self.config['pendulum_thickness']
        cv2.line(image,
                 (cart_x, cart_y - cart_height // 2),
                 (pendulum_end_x, pendulum_end_y),
                 (0, 255, 0), pendulum_thickness)

        # Draw pendulum bob
        cv2.circle(image, (pendulum_end_x, pendulum_end_y),
                   pendulum_thickness * 2, (0, 200, 0), -1)

        # Add text information
        cv2.putText(image, f"Time: {self.time:.2f}s", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Angle: {np.degrees(theta):.1f} deg", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Control: {self.control_history[-1] if self.control_history else 0:.2f}N",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Cart Pos: {x_cart:.2f}m", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image

    def generate_video_frames(self, duration, control_forces=None, save_video=False):
        """
        Generate video frames

        Args:
            duration: simulation duration (seconds)
            control_forces: control force sequence (optional)
            save_video: whether to save as video file

        Returns:
            list of image frames
        """
        num_steps = int(duration * self.config['sampling_rate'])
        frames = []

        print(f"Generating video of {duration}s, total {num_steps} frames")

        for i in range(num_steps):
            # Get control force
            if control_forces is not None and i < len(control_forces):
                control_force = control_forces[i]
            else:
                control_force = 0.0

            # Simulate one step
            self.step(control_force)

            # Generate image
            frame = self.get_current_image()
            frames.append(frame)

            # Show progress
            if i % 100 == 0:
                print(f"  进度: {i}/{num_steps} 帧")

        # Save video
        if save_video and frames:
            self.save_video(frames, "outputs/pendulum_simulation.mp4")

        return frames

    def save_video(self, frames, filename):
        """Save video file"""
        if not frames:
            return

        height, width = frames[0].shape[:2]
        fps = self.config['sampling_rate']

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
        print(f"Video saved to: {filename}")

    def reset(self):
        """Reset the simulator"""
        self.state = np.array([
            self.config['initial_cart_position'],
            self.config['initial_cart_velocity'],
            self.config['initial_angle'],
            self.config['initial_angular_velocity']
        ])
        self.time = 0.0
        self.control_history = []
        self.state_history = []
        self.time_history = []
        self.frame_buffer = []

        print("Simulator reset")

    def get_state_vector(self):
        """Get the state vector"""
        return self.state.copy()

    def get_angle(self):
        """Get the current angle"""
        return self.state[2]

    def get_angular_velocity(self):
        """Get the current angular velocity"""
        return self.state[3]

    def get_cart_position(self):
        """Get the cart position"""
        return self.state[0]