# inverted_pendulum_simulator.py
"""Inverted pendulum on a cart physics and rendering simulator."""
from pathlib import Path

import cv2
import numpy as np


class InvertedPendulumSimulator:
    """Nonlinear cart-pendulum simulator with OpenCV rendering."""

    def __init__(self, config=None):
        self.config = {
            "length": 1.0,
            "mass": 0.1,
            "gravity": 9.81,
            "friction": 0.1,
            "cart_mass": 1.0,
            "cart_friction": 0.05,
            "image_width": 320,
            "image_height": 240,
            "pendulum_thickness": 5,
            "cart_width": 40,
            "cart_height": 20,
            "max_force": 10.0,
            "sampling_rate": 100.0,
            "initial_angle": np.radians(5.0),
            "initial_angular_velocity": 0.0,
            "initial_cart_position": 0.0,
            "initial_cart_velocity": 0.0,
        }
        if config:
            self.config.update(config)

        self.m = self.config["mass"]
        self.M = self.config["cart_mass"]
        self.l = self.config["length"]
        self.g = self.config["gravity"]
        self.b = self.config["friction"]
        self.b_cart = self.config["cart_friction"]
        self.dt = 1.0 / self.config["sampling_rate"]

        self.reset(print_message=False)

        print("Inverted pendulum simulator initialized:")
        print(f"  Pendulum length: {self.l} m, mass: {self.m} kg")
        print(f"  Cart mass: {self.M} kg")
        print(f"  Sampling rate: {self.config['sampling_rate']} Hz")
        print(f"  Initial angle: {np.degrees(self.state[2]):.1f} deg")

    def dynamics(self, t, state, force):
        """Return derivatives for state [x, x_dot, theta, theta_dot]."""
        x, x_dot, theta, theta_dot = state
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        denominator = (self.M + self.m) * self.l ** 2 - (
            self.m ** 2 * self.l ** 2 * cos_theta ** 2
        )
        if abs(denominator) < 1e-9:
            denominator = np.sign(denominator) * 1e-9 if denominator != 0 else 1e-9

        theta_ddot = (
            (self.M + self.m) * self.g * sin_theta
            - self.m * self.l * theta_dot ** 2 * sin_theta * cos_theta
            + (force - self.b_cart * x_dot) * cos_theta
            - self.b * theta_dot
        ) * self.l / denominator

        x_ddot = (
            force
            - self.b_cart * x_dot
            + self.m * self.l * (theta_dot ** 2 * sin_theta - theta_ddot * cos_theta)
        ) / (self.M + self.m)

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=float)

    def step(self, control_force=0.0):
        """Advance the simulator by one RK4 integration step."""
        force = float(np.clip(
            control_force,
            -self.config["max_force"],
            self.config["max_force"],
        ))

        k1 = self.dynamics(self.time, self.state, force)
        k2 = self.dynamics(self.time + self.dt / 2.0, self.state + k1 * self.dt / 2.0, force)
        k3 = self.dynamics(self.time + self.dt / 2.0, self.state + k2 * self.dt / 2.0, force)
        k4 = self.dynamics(self.time + self.dt, self.state + k3 * self.dt, force)

        self.state = self.state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * self.dt / 6.0
        self.time += self.dt

        self.state_history.append(self.state.copy())
        self.control_history.append(force)
        self.time_history.append(self.time)
        return self.state.copy()

    def get_current_image(self):
        """Render the current state as a BGR image."""
        width = self.config["image_width"]
        height = self.config["image_height"]
        image = np.zeros((height, width, 3), dtype=np.uint8)

        x_cart = self.state[0]
        theta = self.state[2]
        ground_y = int(height * 2 / 3)
        scale = self._pixel_scale()
        center_x = width // 2
        cart_x = int(center_x + x_cart * scale)
        cart_y = ground_y
        cart_width = self.config["cart_width"]
        cart_height = self.config["cart_height"]

        pendulum_length_px = int(self.l * scale)
        pivot = (cart_x, cart_y - cart_height // 2)
        pendulum_end = (
            int(pivot[0] + pendulum_length_px * np.sin(theta)),
            int(pivot[1] - pendulum_length_px * np.cos(theta)),
        )

        cv2.line(image, (0, ground_y), (width, ground_y), (200, 200, 200), 2)
        cv2.rectangle(
            image,
            (cart_x - cart_width // 2, cart_y - cart_height),
            (cart_x + cart_width // 2, cart_y),
            (100, 100, 255),
            -1,
        )
        cv2.line(
            image,
            pivot,
            pendulum_end,
            (0, 255, 0),
            self.config["pendulum_thickness"],
        )
        cv2.circle(
            image,
            pendulum_end,
            self.config["pendulum_thickness"] * 2,
            (0, 200, 0),
            -1,
        )

        force = self.control_history[-1] if self.control_history else 0.0
        labels = [
            f"Time: {self.time:.2f}s",
            f"Angle: {np.degrees(theta):.1f} deg",
            f"Control: {force:.2f} N",
            f"Cart Pos: {x_cart:.2f} m",
        ]
        for idx, label in enumerate(labels):
            cv2.putText(
                image,
                label,
                (10, 20 + idx * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return image

    def generate_video_frames(self, duration, control_forces=None, save_video=False):
        """Generate rendered frames while advancing the simulation."""
        num_steps = int(duration * self.config["sampling_rate"])
        frames = []
        print(f"Generating {duration}s simulation, {num_steps} frames")

        for idx in range(num_steps):
            force = 0.0
            if control_forces is not None and idx < len(control_forces):
                force = control_forces[idx]
            self.step(force)
            frames.append(self.get_current_image())
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{num_steps} frames")

        if save_video and frames:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            self.save_video(frames, str(output_dir / "pendulum_simulation.mp4"))
        return frames

    def save_video(self, frames, filename):
        """Save rendered frames to a video file."""
        if not frames:
            return
        height, width = frames[0].shape[:2]
        fps = self.config["sampling_rate"]
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for frame in frames:
            writer.write(frame)
        writer.release()
        print(f"Video saved to: {output_path}")

    def reset(self, print_message=True):
        """Reset simulator state and histories."""
        self.state = np.array([
            self.config["initial_cart_position"],
            self.config["initial_cart_velocity"],
            self.config["initial_angle"],
            self.config["initial_angular_velocity"],
        ], dtype=float)
        self.time = 0.0
        self.control_history = []
        self.state_history = []
        self.time_history = []
        self.frame_buffer = []
        if print_message:
            print("Simulator reset")

    def _pixel_scale(self):
        width = self.config["image_width"]
        height = self.config["image_height"]
        ground_y = int(height * 2 / 3)
        horizontal_scale = width / (4 * self.l)
        vertical_scale = (ground_y - 20) / (2 * self.l)
        return min(horizontal_scale, vertical_scale)

    def get_cart_x_pixel(self):
        """Return the cart pivot x-coordinate in image pixels."""
        return int(self.config["image_width"] // 2 + self.state[0] * self._pixel_scale())

    def get_pendulum_length_pixels(self):
        """Return the rendered pendulum length in pixels."""
        return int(self.l * self._pixel_scale())

    def get_state_vector(self):
        """Return [cart position, cart velocity, angle, angular velocity]."""
        return self.state.copy()

    def get_angle(self):
        return self.state[2]

    def get_angular_velocity(self):
        return self.state[3]

    def get_cart_position(self):
        return self.state[0]


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the standalone inverted-pendulum simulator."
    )
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Simulation duration in seconds.")
    parser.add_argument("--sampling-rate", type=float, default=100.0,
                        help="Physics sampling rate in Hz.")
    parser.add_argument("--initial-angle", type=float, default=5.0,
                        help="Initial pendulum angle in degrees.")
    parser.add_argument("--force", type=float, default=0.0,
                        help="Constant force applied to the cart in newtons.")
    parser.add_argument("--width", type=int, default=320,
                        help="Rendered image width.")
    parser.add_argument("--height", type=int, default=240,
                        help="Rendered image height.")
    parser.add_argument("--headless", action="store_true",
                        help="Run without OpenCV windows.")
    parser.add_argument("--save-video", action="store_true",
                        help="Save outputs/pendulum_simulation.mp4.")
    return parser.parse_args()


def main():
    args = parse_args()
    simulator = InvertedPendulumSimulator({
        "image_width": args.width,
        "image_height": args.height,
        "sampling_rate": args.sampling_rate,
        "initial_angle": np.radians(args.initial_angle),
    })

    num_steps = int(args.duration * args.sampling_rate)
    frames = []
    report_interval = max(1, int(args.sampling_rate))

    for step_idx in range(num_steps):
        simulator.step(args.force)
        frame = simulator.get_current_image()
        if args.save_video:
            frames.append(frame)
        if not args.headless:
            cv2.imshow("Inverted Pendulum Simulator", frame)
            if cv2.waitKey(max(1, int(1000 / args.sampling_rate))) & 0xFF == ord("q"):
                break
        if step_idx % report_interval == 0:
            print(
                f"  t={simulator.time:.2f}s "
                f"angle={np.degrees(simulator.get_angle()):+.2f} deg "
                f"cart={simulator.get_cart_position():+.3f} m"
            )

    if args.save_video and frames:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        simulator.save_video(frames, str(output_dir / "pendulum_simulation.mp4"))

    cv2.destroyAllWindows()
    print("Final state:", simulator.get_state_vector())


if __name__ == "__main__":
    main()
