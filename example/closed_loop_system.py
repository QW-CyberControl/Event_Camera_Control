# closed_loop_system.py
"""Closed-loop inverted-pendulum control with event-camera state estimation."""
from pathlib import Path
import sys
import time

import cv2
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from inverted_pendulum_simulator import InvertedPendulumSimulator
from integrated_event_camera import IntegratedEventCamera
from event_based_estimator import EventBasedEstimator
from simple_state_estimator import SimpleStateEstimator
from pendulum_controller import PendulumController


class ClosedLoopSystem:
    """Coordinate pendulum rendering, events, estimation, and control."""

    def __init__(self, config=None):
        self.config = {
            "simulation_duration": 20.0,
            "real_time_factor": 0.0,
            "video_width": 480,
            "video_height": 360,
            "show_display": False,
            "display_events": False,
            "save_video": False,
            "estimator": "event",       # event or ground_truth
            "controller": "LQR",
            "initial_angle_deg": 8.0,
            "initial_cart_position": 0.0,
            "initial_cart_velocity": 0.0,
            "initial_angular_velocity": 0.0,
            "output_dir": "outputs",
            "seed": 7,
        }
        if config:
            self.config.update(config)
        np.random.seed(self.config["seed"])

        print("=" * 60)
        print("Event-Camera Inverted-Pendulum Control")
        print("=" * 60)

        self.pendulum = InvertedPendulumSimulator({
            "image_width": self.config["video_width"],
            "image_height": self.config["video_height"],
            "initial_angle": np.radians(self.config["initial_angle_deg"]),
            "initial_cart_position": self.config["initial_cart_position"],
            "initial_cart_velocity": self.config["initial_cart_velocity"],
            "initial_angular_velocity": self.config["initial_angular_velocity"],
            "sampling_rate": 200.0,
        })
        pendulum_length_px = self.pendulum.get_pendulum_length_pixels()

        self.event_camera = IntegratedEventCamera(
            self.config["video_width"],
            self.config["video_height"],
            {
                "dt": 10000,
                "th_pos": 0.18,
                "th_neg": 0.18,
                "th_noise": 0.01,
                "bgnp": 0.001,
                "bgnn": 0.001,
                "display_events": self.config["display_events"],
            },
            tx_config={
                "enabled": True,
                "bandwidth": 600,
                "packet_loss": 0.02,
                "latency_us": 200,
                "jitter_us": 30,
            },
            filter_config={
                "roi_mode": "circle",
                "pendulum_length_pixels": pendulum_length_px,
                "circle_radius_scale": 1.35,
                "refractory_us": 250,
                "min_events_per_frame": 0,
            },
        )

        if self.config["estimator"] == "ground_truth":
            self.estimator = SimpleStateEstimator(
                self.config["video_width"],
                self.config["video_height"],
                {
                    "use_ground_truth": True,
                    "angle_noise_std": 0.002,
                    "vel_noise_std": 0.02,
                    "cart_noise_std": 0.005,
                    "cart_vel_noise_std": 0.02,
                },
            )
            self.estimator.set_ground_truth_callback(self._ground_truth_state)
        else:
            estimator_config = {
                "initial_angle": np.radians(self.config["initial_angle_deg"]),
                "use_initial_angle_hint": abs(self.config["initial_angle_deg"]) > 90.0,
            }
            if self.config["controller"] == "SwingUpLQR":
                estimator_config.update({
                    "y_min": 5,
                    "y_max": self.config["video_height"] - 5,
                    "cart_y_min": int(self.config["video_height"] * 2 / 3) - 25,
                    "cart_y_max": int(self.config["video_height"] * 2 / 3) + 5,
                    "recent_time_fraction": 0.35,
                    "min_events": 14,
                    "confidence_min": 0.08,
                })
            self.estimator = EventBasedEstimator(
                self.config["video_width"],
                self.config["video_height"],
                estimator_config,
            )

        self.controller = PendulumController({
            "controller_type": self.config["controller"],
            "K_cart_pos": -3.1623,
            "K_cart_vel": -5.0317,
            "K_angle": 47.7304,
            "K_angle_vel": 13.979,
            "max_force": 18.0 if self.config["controller"] == "SwingUpLQR" else 12.0,
            "sampling_rate": 100.0,
            "pendulum_mass": self.pendulum.m,
            "pendulum_length": self.pendulum.l,
            "gravity": self.pendulum.g,
            "swingup_gain": 12.0,
            "swingup_cart_gain": 1.0,
            "swingup_cart_vel_gain": 1.6,
        })

        self.running = False
        self.simulation_time = 0.0
        self.frame_count = 0
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(exist_ok=True)
        self.log = self._empty_log()
        self.video_writer = None

    def run_simulation(self):
        dt_physics = 1.0 / self.pendulum.config["sampling_rate"]
        dt_control = self.event_camera.config["dt"] * 1e-6
        physics_steps = max(1, int(round(dt_control / dt_physics)))
        total_frames = int(self.config["simulation_duration"] / dt_control)

        print(f"Estimator: {self.config['estimator']}")
        print(f"Control rate: {1 / dt_control:.1f} Hz")
        print(f"Physics rate: {1 / dt_physics:.1f} Hz")
        print(f"Frames: {total_frames}")

        self._reset_all()
        frame = self.pendulum.get_current_image()
        self.event_camera.init_with_frame(frame)

        self.running = True
        started = time.time()
        force = 0.0
        self._open_video_writer(dt_control)

        for idx in range(total_frames):
            self.simulation_time = idx * dt_control

            frame = self.pendulum.get_current_image()
            cart_x_pixel = self.pendulum.get_cart_x_pixel()
            events = self.event_camera.process_frame(frame, cart_x_pixel=cart_x_pixel)

            angle, angle_vel, valid, cart_pos, cart_vel = self.estimator.estimate_from_events(
                events,
                self.event_camera.current_time_us,
            )
            if valid or self.config["controller"] == "SwingUpLQR":
                force = self.controller.compute_control(
                    angle,
                    angle_vel,
                    self.simulation_time,
                    cart_pos=cart_pos,
                    cart_velocity=cart_vel,
                )
            else:
                force *= 0.95

            for _ in range(physics_steps):
                self.pendulum.step(force)

            self._log_step(angle, angle_vel, cart_pos, cart_vel, valid, force, events)
            self.frame_count += 1

            if self.config["show_display"] or self.config["save_video"]:
                status_frame = self._make_status_frame(
                    frame, angle, angle_vel, cart_pos, cart_vel, valid, force, events.i
                )
                if self.video_writer is not None:
                    self.video_writer.write(status_frame)
            if self.config["show_display"]:
                cv2.imshow("Event-Camera Closed-Loop Control", status_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            self._sleep_if_needed(started, idx + 1, dt_control)

        self.running = False
        elapsed = time.time() - started
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        cv2.destroyAllWindows()
        return self._report(elapsed)

    def _ground_truth_state(self):
        return (
            self.pendulum.get_angle(),
            self.pendulum.get_angular_velocity(),
            self.pendulum.get_cart_position(),
            self.pendulum.state[1],
        )

    def _reset_all(self):
        self.pendulum.reset()
        self.event_camera.reset()
        self.estimator.reset()
        self.controller.reset()
        self.log = self._empty_log()
        self.frame_count = 0

    @staticmethod
    def _empty_log():
        return {
            "t": [],
            "true_angle": [],
            "est_angle": [],
            "true_angle_vel": [],
            "est_angle_vel": [],
            "true_cart": [],
            "est_cart": [],
            "true_cart_vel": [],
            "est_cart_vel": [],
            "force": [],
            "events": [],
            "valid": [],
            "confidence": [],
            "control_mode": [],
        }

    def _log_step(self, angle, angle_vel, cart_pos, cart_vel, valid, force, events):
        truth = self._ground_truth_state()
        self.log["t"].append(self.simulation_time)
        self.log["true_angle"].append(truth[0])
        self.log["est_angle"].append(angle)
        self.log["true_angle_vel"].append(truth[1])
        self.log["est_angle_vel"].append(angle_vel)
        self.log["true_cart"].append(truth[2])
        self.log["est_cart"].append(cart_pos)
        self.log["true_cart_vel"].append(truth[3])
        self.log["est_cart_vel"].append(cart_vel)
        self.log["force"].append(force)
        self.log["events"].append(events.i if events else 0)
        self.log["valid"].append(bool(valid))
        self.log["confidence"].append(getattr(self.estimator, "confidence", 1.0 if valid else 0.0))
        self.log["control_mode"].append(getattr(self.controller, "current_mode", self.config["controller"]))

    def _make_status_frame(self, frame, angle, angle_vel, cart_pos, cart_vel, valid, force, events_count):
        image = frame.copy()
        true_angle, true_angle_vel, true_cart, true_cart_vel = self._ground_truth_state()
        true_angle = self._wrap_angle(true_angle)
        angle = self._wrap_angle(angle)
        y0 = 108
        lines = [
            "STATE                 TRUE        EST",
            f"theta deg        {np.degrees(true_angle):+8.2f} {np.degrees(angle):+8.2f}",
            f"theta_dot deg/s  {np.degrees(true_angle_vel):+8.2f} {np.degrees(angle_vel):+8.2f}",
            f"cart x m         {true_cart:+8.3f} {cart_pos:+8.3f}",
            f"cart v m/s       {true_cart_vel:+8.3f} {cart_vel:+8.3f}",
            f"force: {force:+7.2f} N  mode: {getattr(self.controller, 'current_mode', '-')}",
            f"events: {events_count}  valid: {valid}",
        ]
        for idx, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (10, y0 + idx * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
        return image

    def _open_video_writer(self, dt_control):
        if not self.config["save_video"]:
            self.video_writer = None
            return
        output_path = self.output_dir / "closed_loop_swingup_demo.mp4"
        fps = max(1.0, min(60.0, 1.0 / dt_control))
        size = (self.config["video_width"], self.config["video_height"])
        self.video_writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            size,
        )
        print(f"Saving display video to: {output_path}")

    def _sleep_if_needed(self, started, completed_frames, dt_control):
        factor = float(self.config["real_time_factor"])
        if factor <= 0:
            return
        target_elapsed = completed_frames * dt_control / factor
        remaining = target_elapsed - (time.time() - started)
        if remaining > 0:
            time.sleep(remaining)

    def _report(self, elapsed):
        arrays = {name: np.array(values) for name, values in self.log.items()}
        np.savez(self.output_dir / "closed_loop_event_lqr_result.npz", **arrays)

        true_angle_wrapped = np.array([self._wrap_angle(v) for v in arrays["true_angle"]])
        est_angle_wrapped = np.array([self._wrap_angle(v) for v in arrays["est_angle"]])
        angle_deg = np.degrees(true_angle_wrapped)
        est_angle_deg = np.degrees(est_angle_wrapped)
        cart = arrays["true_cart"]
        force = arrays["force"]
        valid_ratio = float(np.mean(arrays["valid"])) if len(arrays["valid"]) else 0.0

        steady_n = max(1, len(angle_deg) // 3)
        steady_angle = angle_deg[-steady_n:]
        steady_cart = cart[-steady_n:]
        steady_force = force[-steady_n:]
        modes = arrays["control_mode"] if "control_mode" in arrays else np.array([])
        mode_counts = {
            str(mode): int(np.sum(modes == mode))
            for mode in np.unique(modes)
        } if len(modes) else {}

        summary = {
            "frames": self.frame_count,
            "wall_time_s": elapsed,
            "final_angle_deg": float(angle_deg[-1]),
            "final_cart_m": float(cart[-1]),
            "steady_rms_angle_deg": float(np.sqrt(np.mean(steady_angle ** 2))),
            "steady_max_angle_deg": float(np.max(np.abs(steady_angle))),
            "steady_max_cart_m": float(np.max(np.abs(steady_cart))),
            "mean_abs_force_n": float(np.mean(np.abs(steady_force))),
            "mean_abs_angle_error_deg": float(np.mean(np.abs(np.degrees(
                [self._wrap_angle(e - t) for e, t in zip(est_angle_wrapped, true_angle_wrapped)]
            )))),
            "valid_ratio": valid_ratio,
            "event_stats": self.event_camera.get_event_statistics(),
            "mode_counts": mode_counts,
        }
        summary["stable"] = (
            summary["steady_rms_angle_deg"] <= 5.0 and
            summary["steady_max_angle_deg"] <= 12.0 and
            summary["steady_max_cart_m"] <= 0.5 and
            summary["valid_ratio"] >= 0.8
        )

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  frames: {summary['frames']}")
        print(f"  wall time: {summary['wall_time_s']:.2f}s")
        print(f"  final angle: {summary['final_angle_deg']:+.2f} deg")
        print(f"  final cart: {summary['final_cart_m']:+.3f} m")
        print(f"  steady RMS angle: {summary['steady_rms_angle_deg']:.2f} deg")
        print(f"  steady max angle: {summary['steady_max_angle_deg']:.2f} deg")
        print(f"  steady max cart: {summary['steady_max_cart_m']:.3f} m")
        print(f"  mean |angle error|: {summary['mean_abs_angle_error_deg']:.2f} deg")
        print(f"  valid estimates: {summary['valid_ratio'] * 100:.1f}%")
        print(f"  control modes: {summary['mode_counts']}")
        print(f"  stable: {summary['stable']}")
        print(f"  result file: {self.output_dir / 'closed_loop_event_lqr_result.npz'}")
        return summary

    @staticmethod
    def _wrap_angle(angle):
        return float((angle + np.pi) % (2 * np.pi) - np.pi)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run event-camera closed-loop inverted-pendulum control."
    )
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Simulation duration in seconds.")
    parser.add_argument("--estimator", choices=["event", "ground_truth"],
                        default="event",
                        help="Use event-based estimation or noisy ground-truth debug mode.")
    parser.add_argument("--controller", choices=["LQR", "SwingUpLQR"],
                        default="LQR",
                        help="Use LQR near upright or hybrid swing-up plus LQR.")
    parser.add_argument("--initial-angle", type=float, default=8.0,
                        help="Initial pendulum angle in degrees. 0 is upright, 180 is downward.")
    parser.add_argument("--headless", action="store_true",
                        help="Disable OpenCV windows for batch experiments.")
    parser.add_argument("--display-events", action="store_true",
                        help="Show the filtered event-camera stream.")
    parser.add_argument("--save-video", action="store_true",
                        help="Save the rendered status display to outputs/closed_loop_swingup_demo.mp4.")
    parser.add_argument("--real-time", type=float, default=0.0,
                        help="Real-time factor. 0 runs as fast as possible.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for channel noise and estimator noise.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = {
        "simulation_duration": args.duration,
        "show_display": not args.headless,
        "display_events": args.display_events,
        "save_video": args.save_video,
        "estimator": args.estimator,
        "controller": args.controller,
        "initial_angle_deg": args.initial_angle,
        "real_time_factor": args.real_time,
        "seed": args.seed,
    }
    system = ClosedLoopSystem(config)
    system.run_simulation()


if __name__ == "__main__":
    main()
