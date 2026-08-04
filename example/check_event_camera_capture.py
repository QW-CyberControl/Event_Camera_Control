# check_event_camera_capture.py
"""Check whether the DVS pipeline produces events from pendulum motion."""
from pathlib import Path
import sys

import cv2
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from inverted_pendulum_simulator import InvertedPendulumSimulator
from integrated_event_camera import IntegratedEventCamera


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Feed pendulum frames into the event camera and report event statistics."
    )
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Simulation duration in seconds.")
    parser.add_argument("--width", type=int, default=320,
                        help="Rendered image width.")
    parser.add_argument("--height", type=int, default=240,
                        help="Rendered image height.")
    parser.add_argument("--initial-angle", type=float, default=8.0,
                        help="Initial pendulum angle in degrees.")
    parser.add_argument("--force", type=float, default=0.0,
                        help="Constant force applied during the check.")
    parser.add_argument("--headless", action="store_true",
                        help="Disable OpenCV preview windows.")
    parser.add_argument("--save-preview", action="store_true",
                        help="Save outputs/event_camera_capture_preview.mp4.")
    parser.add_argument("--static", action="store_true",
                        help="Keep feeding the first frame as a no-motion baseline.")
    parser.add_argument("--background-noise", type=float, default=0.001,
                        help="Background noise frequency passed to DVS bgnp/bgnn.")
    return parser.parse_args()


def make_event_image(events, width, height):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if events is None or events.i == 0:
        return image

    x = events.get_x()
    y = events.get_y()
    p = events.get_p()
    on = p == 1
    off = ~on
    image[y[on], x[on]] = (0, 255, 0)
    image[y[off], x[off]] = (0, 0, 255)
    return image


def main():
    args = parse_args()
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

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
        tx_config={"enabled": False},
        filter_config={
            "roi_mode": "circle",
            "pendulum_length_pixels": pendulum.get_pendulum_length_pixels(),
            "circle_radius_scale": 1.35,
            "refractory_us": 0,
        },
    )

    first_frame = pendulum.get_current_image()
    event_camera.init_with_frame(first_frame)

    frames = int(args.duration * pendulum.config["sampling_rate"])
    counts = []
    polarities = []
    ranges = []
    preview_frames = []
    last_ts = -1
    timestamps_ok = True

    for idx in range(frames):
        if args.static:
            frame = first_frame
        else:
            pendulum.step(args.force)
            frame = pendulum.get_current_image()
        events = event_camera.process_frame(frame, cart_x_pixel=pendulum.get_cart_x_pixel())
        counts.append(events.i)

        if events.i > 0:
            x = events.get_x()
            y = events.get_y()
            p = events.get_p()
            ts = events.get_ts()
            polarities.append((int(np.sum(p == 1)), int(np.sum(p == 0))))
            ranges.append((int(x.min()), int(x.max()), int(y.min()), int(y.max())))
            timestamps_ok = timestamps_ok and int(ts.min()) >= last_ts
            last_ts = int(ts.max())

        event_image = make_event_image(events, args.width, args.height)
        preview = np.hstack([frame, event_image])
        cv2.putText(preview, f"events: {events.i}", (args.width + 10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if args.save_preview:
            preview_frames.append(preview)
        if not args.headless:
            cv2.imshow("Pendulum frame | Event image", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

    counts_np = np.array(counts)
    summary = {
        "frames": len(counts),
        "total_events": int(counts_np.sum()),
        "nonzero_frames": int(np.sum(counts_np > 0)),
        "max_events_frame": int(counts_np.max()) if len(counts_np) else 0,
        "mean_events_frame": float(counts_np.mean()) if len(counts_np) else 0.0,
        "timestamps_monotonic_blocks": bool(timestamps_ok),
        "sample_ranges": ranges[:5],
        "sample_polarities_on_off": polarities[:5],
        "camera_stats": event_camera.get_event_statistics(),
    }

    np.savez(output_dir / "event_camera_capture_check.npz", counts=counts_np)

    if args.save_preview and preview_frames:
        height, width = preview_frames[0].shape[:2]
        path = output_dir / "event_camera_capture_preview.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
        for frame in preview_frames:
            writer.write(frame)
        writer.release()
        print(f"Preview video saved to: {path}")

    print("EVENT CAMERA CAPTURE CHECK")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
