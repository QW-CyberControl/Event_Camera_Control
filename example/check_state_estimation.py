# check_state_estimation.py
"""Evaluate event-based state estimates against simulator ground truth."""
from pathlib import Path
import sys

import cv2
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from inverted_pendulum_simulator import InvertedPendulumSimulator
from integrated_event_camera import IntegratedEventCamera
from event_based_estimator import EventBasedEstimator


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Check event transmission plus state estimation accuracy."
    )
    parser.add_argument("--duration", type=float, default=4.0,
                        help="Simulation duration in seconds.")
    parser.add_argument("--width", type=int, default=320,
                        help="Rendered image width.")
    parser.add_argument("--height", type=int, default=240,
                        help="Rendered image height.")
    parser.add_argument("--initial-angle", type=float, default=8.0,
                        help="Initial pendulum angle in degrees.")
    parser.add_argument("--force", type=float, default=0.0,
                        help="Constant force applied to the cart.")
    parser.add_argument("--bandwidth", type=int, default=600,
                        help="Transmission bandwidth limit in events/frame; -1 disables.")
    parser.add_argument("--packet-loss", type=float, default=0.02,
                        help="Independent event packet-loss probability.")
    parser.add_argument("--latency-us", type=int, default=200,
                        help="Fixed transmission latency in microseconds.")
    parser.add_argument("--jitter-us", type=int, default=30,
                        help="Timestamp jitter in microseconds.")
    parser.add_argument("--background-noise", type=float, default=0.0,
                        help="DVS background noise frequency. 0 disables it.")
    parser.add_argument("--headless", action="store_true",
                        help="Disable OpenCV preview.")
    parser.add_argument("--max-eval-angle", type=float, default=60.0,
                        help="Only evaluate frames with |true angle| below this value.")
    parser.add_argument("--angle-threshold", type=float, default=12.0,
                        help="PASS threshold for angle MAE in degrees.")
    parser.add_argument("--cart-threshold", type=float, default=0.25,
                        help="PASS threshold for cart-position MAE in meters.")
    return parser.parse_args()


def mae(values):
    return float(np.mean(np.abs(values))) if len(values) else float("nan")


def rmse(values):
    return float(np.sqrt(np.mean(values * values))) if len(values) else float("nan")


def draw_preview(frame, truth, estimate, valid, events_count):
    image = frame.copy()
    lines = [
        f"truth angle: {np.degrees(truth[2]):+6.2f} deg",
        f"est angle:   {np.degrees(estimate[0]):+6.2f} deg",
        f"truth cart:  {truth[0]:+6.3f} m",
        f"est cart:    {estimate[3]:+6.3f} m",
        f"events: {events_count} valid: {valid}",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(image, line, (10, 120 + idx * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return image


def main():
    args = parse_args()
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    np.random.seed(11)

    pendulum = InvertedPendulumSimulator({
        "image_width": args.width,
        "image_height": args.height,
        "sampling_rate": 100.0,
        "initial_angle": np.radians(args.initial_angle),
    })
    event_camera = IntegratedEventCamera(
        args.width,
        args.height,
        {
            "dt": 10000,
            "th_pos": 0.18,
            "th_neg": 0.18,
            "th_noise": 0.01,
            "bgnp": args.background_noise,
            "bgnn": args.background_noise,
            "display_events": False,
        },
        tx_config={
            "enabled": True,
            "bandwidth": args.bandwidth,
            "packet_loss": args.packet_loss,
            "latency_us": args.latency_us,
            "jitter_us": args.jitter_us,
        },
        filter_config={
            "roi_mode": "circle",
            "pendulum_length_pixels": pendulum.get_pendulum_length_pixels(),
            "circle_radius_scale": 1.35,
            "refractory_us": 250,
        },
    )
    estimator = EventBasedEstimator(args.width, args.height)

    first_frame = pendulum.get_current_image()
    event_camera.init_with_frame(first_frame)

    frames = int(args.duration * pendulum.config["sampling_rate"])
    log = {
        "t": [],
        "true_angle": [],
        "est_angle": [],
        "true_angle_vel": [],
        "est_angle_vel": [],
        "true_cart": [],
        "est_cart": [],
        "true_cart_vel": [],
        "est_cart_vel": [],
        "valid": [],
        "events": [],
        "confidence": [],
    }

    for idx in range(frames):
        pendulum.step(args.force)
        frame = pendulum.get_current_image()
        events = event_camera.process_frame(frame, cart_x_pixel=pendulum.get_cart_x_pixel())
        angle, angle_vel, valid, cart_pos, cart_vel = estimator.estimate_from_events(
            events,
            event_camera.current_time_us,
        )

        true_state = pendulum.get_state_vector()
        log["t"].append(pendulum.time)
        log["true_cart"].append(true_state[0])
        log["true_cart_vel"].append(true_state[1])
        log["true_angle"].append(true_state[2])
        log["true_angle_vel"].append(true_state[3])
        log["est_angle"].append(angle)
        log["est_angle_vel"].append(angle_vel)
        log["est_cart"].append(cart_pos)
        log["est_cart_vel"].append(cart_vel)
        log["valid"].append(valid)
        log["events"].append(events.i)
        log["confidence"].append(estimator.confidence)

        if not args.headless:
            preview = draw_preview(
                frame,
                true_state,
                (angle, angle_vel, valid, cart_pos, cart_vel),
                valid,
                events.i,
            )
            cv2.imshow("State Estimation Check", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

    arrays = {key: np.array(value) for key, value in log.items()}
    valid = arrays["valid"].astype(bool)
    warmup = arrays["t"] > 0.25
    operating_range = np.abs(np.degrees(arrays["true_angle"])) <= args.max_eval_angle
    mask = valid & warmup & operating_range

    if np.sum(mask) == 0:
        raise RuntimeError("No valid event-based estimates were produced.")

    angle_error = arrays["est_angle"][mask] - arrays["true_angle"][mask]
    angle_vel_error = arrays["est_angle_vel"][mask] - arrays["true_angle_vel"][mask]
    cart_error = arrays["est_cart"][mask] - arrays["true_cart"][mask]
    cart_vel_error = arrays["est_cart_vel"][mask] - arrays["true_cart_vel"][mask]

    summary = {
        "frames": int(len(arrays["t"])),
        "evaluated_frames": int(np.sum(mask)),
        "valid_ratio": float(np.mean(valid)),
        "operating_range_ratio": float(np.mean(operating_range)),
        "mean_events_frame": float(np.mean(arrays["events"])),
        "angle_mae_deg": float(np.degrees(mae(angle_error))),
        "angle_rmse_deg": float(np.degrees(rmse(angle_error))),
        "angle_vel_mae_deg_s": float(np.degrees(mae(angle_vel_error))),
        "cart_mae_m": mae(cart_error),
        "cart_rmse_m": rmse(cart_error),
        "cart_vel_mae_m_s": mae(cart_vel_error),
        "mean_confidence": float(np.mean(arrays["confidence"][mask])),
        "event_stats": event_camera.get_event_statistics(),
    }
    summary["pass"] = (
        summary["angle_mae_deg"] <= args.angle_threshold and
        summary["cart_mae_m"] <= args.cart_threshold
    )

    np.savez(output_dir / "state_estimation_check.npz", **arrays)
    print("STATE ESTIMATION CHECK")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
