# integrated_event_camera.py
"""Integrated event camera pipeline for the inverted-pendulum simulator.

Pipeline:
    rendered frame -> DVS sensor -> transmission channel -> event filter -> events
"""
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dvs_sensor import DvsSensor
from event_buffer import EventBuffer
from event_display import EventDisplay
from event_transmission import EventTransmission
from event_filter import EventFilter


class IntegratedEventCamera:
    """DVS sensor plus transmission and filtering stages."""

    def __init__(self, width, height, config=None, tx_config=None, filter_config=None):
        self.width = width
        self.height = height

        self.config = {
            "th_pos": 0.4,
            "th_neg": 0.4,
            "th_noise": 0.01,
            "lat": 100,
            "tau": 40,
            "jit": 10,
            "bgnp": 0.1,
            "bgnn": 0.01,
            "ref": 100,
            "dt": 1000,
            "display_events": False,
        }
        if config:
            self.config.update(config)

        self.dvs = self._make_sensor()
        self.transmission = EventTransmission(width, height, tx_config)
        self.event_filter = EventFilter(width, height, filter_config)
        self.event_buffer = EventBuffer(1000)

        self.event_display = None
        if self.config["display_events"]:
            self.event_display = EventDisplay(
                "Event Camera Output",
                width,
                height,
                self.config["dt"],
                render=1,
            )

        self.current_time_us = 0
        self.frame_count = 0
        self.event_count = 0
        self.event_rate_history = []

        print(f"Integrated event camera initialized: {width}x{height}")
        print("  Pipeline: DVS -> Transmission -> Filter -> Output")

    def _make_sensor(self):
        dvs = DvsSensor("IntegratedDVS")
        dvs.initCamera(
            self.width,
            self.height,
            lat=self.config["lat"],
            jit=self.config["jit"],
            ref=self.config["ref"],
            tau=self.config["tau"],
            th_pos=self.config["th_pos"],
            th_neg=self.config["th_neg"],
            th_noise=self.config["th_noise"],
            bgnp=self.config["bgnp"],
            bgnn=self.config["bgnn"],
        )
        return dvs

    def init_with_frame(self, frame):
        """Initialize the DVS reference image from a BGR frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dvs_input = gray / 255.0 * 1e4
        self.dvs.init_image(dvs_input)
        print("DVS initialized with first frame")

    def process_frame(self, frame, dt_us=None, cart_x_pixel=None):
        """Process one rendered frame and return filtered events."""
        if dt_us is None:
            dt_us = self.config["dt"]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dvs_input = gray / 255.0 * 1e4

        raw_events = self.dvs.update(dvs_input, dt_us)
        tx_events = self.transmission.transmit(raw_events)

        if cart_x_pixel is not None:
            self.event_filter.set_cart_x(cart_x_pixel)
        filtered_events = self.event_filter.filter(tx_events, self.current_time_us)

        self.current_time_us += dt_us
        self.frame_count += 1
        self.event_count += filtered_events.i
        self.event_rate_history.append(filtered_events.i / (dt_us * 1e-6))

        if self.event_display is not None:
            self.event_display.update(filtered_events, dt_us)

        return filtered_events

    def get_event_statistics(self):
        """Return event-rate statistics plus transmission and filter diagnostics."""
        if self.event_rate_history:
            base = {
                "total_events": self.event_count,
                "average_rate": float(np.mean(self.event_rate_history)),
                "frame_count": self.frame_count,
                "current_time_us": self.current_time_us,
                "current_time_s": self.current_time_us * 1e-6,
            }
        else:
            base = {
                "total_events": 0,
                "average_rate": 0.0,
                "frame_count": 0,
                "current_time_us": self.current_time_us,
                "current_time_s": self.current_time_us * 1e-6,
            }

        base["transmission"] = self.transmission.get_statistics()
        base["filter"] = self.event_filter.get_statistics()
        return base

    def reset(self):
        """Reset DVS state, pipeline modules, display buffers, and statistics."""
        self.dvs = self._make_sensor()
        self.transmission.reset()
        self.event_filter.reset()
        if self.event_display is not None:
            self.event_display.reset()

        self.current_time_us = 0
        self.frame_count = 0
        self.event_count = 0
        self.event_rate_history = []

        print("Event camera reset")
