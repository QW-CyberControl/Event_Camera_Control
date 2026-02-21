# integrated_event_camera.py
"""
Integrated event camera simulator, working together with the inverted pendulum simulator"""
import numpy as np
import cv2
import sys

sys.path.append("../src")

from dvs_sensor import DvsSensor
from event_buffer import EventBuffer
from event_display import EventDisplay


class IntegratedEventCamera:
    """Integrated event camera simulator"""

    def __init__(self, width, height, config=None):
        """
        Initialize the event camera simulator

        Args:
            width: image width
            height: image height
            config: DVS configuration parameters
        """
        self.width = width
        self.height = height

        # DVS default configuration
        self.config = {
            'th_pos': 0.4,
            'th_neg': 0.4,
            'th_noise': 0.01,
            'lat': 100,
            'tau': 40,
            'jit': 10,
            'bgnp': 0.1,
            'bgnn': 0.01,
            'ref': 100,
            'dt': 1000,  # microseconds
        }

        if config:
            self.config.update(config)

        # Initialize DVS sensor
        self.dvs = DvsSensor("IntegratedDVS")
        self.dvs.initCamera(
            width, height,
            lat=self.config['lat'],
            jit=self.config['jit'],
            ref=self.config['ref'],
            tau=self.config['tau'],
            th_pos=self.config['th_pos'],
            th_neg=self.config['th_neg'],
            th_noise=self.config['th_noise'],
            bgnp=self.config['bgnp'],
            bgnn=self.config['bgnn']
        )

        # Event buffer
        self.event_buffer = EventBuffer(1000)

        # Event display
        render_timesurface = 1
        self.event_display = EventDisplay(
            "Event Camera Output",
            width, height,
            self.config['dt'],
            render_timesurface
        )

        # Time management
        self.current_time_us = 0
        self.frame_count = 0

        # Event statistics
        self.event_count = 0
        self.event_rate_history = []

        print(f"Integrated event camera initialized: {width}x{height}")

    def init_with_frame(self, frame):
        """
        Initialize the DVS with the first frame

        Args:
            frame: first frame image (BGR format)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert to DVS input format
        dvs_input = gray / 255.0 * 1e4

        # Initialize DVS
        self.dvs.init_image(dvs_input)

        print("DVS initialized with first frame")

    def process_frame(self, frame, dt_us=None):
        """
        Process one frame and generate events

        Args:
            frame: current frame image (BGR format)
            dt_us: time interval (microseconds), if None use dt from config

        Returns:
            event buffer
        """
        if dt_us is None:
            dt_us = self.config['dt']

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert to DVS input format
        dvs_input = gray / 255.0 * 1e4

        # Generate events
        events = self.dvs.update(dvs_input, dt_us)

        # Update time
        self.current_time_us += dt_us
        self.frame_count += 1

        # Update event statistics
        self.event_count += events.i
        self.event_rate_history.append(events.i / (dt_us * 1e-6))  # vents per second

        # Display events
        self.event_display.update(events, dt_us)

        return events

    def get_event_statistics(self):
        """Get event statistics"""
        if not self.event_rate_history:
            return {
                'total_events': 0,
                'average_rate': 0,
                'frame_count': 0
            }

        avg_rate = np.mean(self.event_rate_history)

        return {
            'total_events': self.event_count,
            'average_rate': avg_rate,
            'frame_count': self.frame_count,
            'current_time_us': self.current_time_us,
            'current_time_s': self.current_time_us * 1e-6
        }

    def reset(self):
        """Reset the event camera"""
        # Re‑initialize DVS
        self.dvs = DvsSensor("IntegratedDVS")
        self.dvs.initCamera(
            self.width, self.height,
            lat=self.config['lat'],
            jit=self.config['jit'],
            ref=self.config['ref'],
            tau=self.config['tau'],
            th_pos=self.config['th_pos'],
            th_neg=self.config['th_neg'],
            th_noise=self.config['th_noise'],
            bgnp=self.config['bgnp'],
            bgnn=self.config['bgnn']
        )

        # Reset state
        self.current_time_us = 0
        self.frame_count = 0
        self.event_count = 0
        self.event_rate_history = []

        print("Event camera reset")
