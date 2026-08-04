# run_event_control.py
"""
Event-driven closed-loop control for inverted pendulum.
不使用任何状态估计器，事件相机信号→直接→控制力

运行方式：
  cd example/
  python run_event_control.py
"""
import numpy as np
import cv2
import time
import os
import sys

sys.path.insert(0, '.')

from inverted_pendulum_simulator import InvertedPendulumSimulator
from integrated_event_camera import IntegratedEventCamera
from event_controller import EventDrivenController


class EventDrivenSystem:
    def __init__(self, config=None):
        self.config = {
            'simulation_duration': 25.0,
            'real_time_factor': 1.0,
            'video_width': 480,
            'video_height': 360,
            'show_display': True,
        }
        if config:
            self.config.update(config)

        print("=" * 60)
        print("Event-Driven Control System")
        print("No state estimator | Events -> Force directly")
        print("=" * 60)

        # 1. 倒立摆（初始角度 8°，有明显扰动）
        pendulum_cfg = {
            'image_width': self.config['video_width'],
            'image_height': self.config['video_height'],
            'initial_angle': np.radians(8.0),
        }
        self.pendulum = InvertedPendulumSimulator(pendulum_cfg)
        px = self.pendulum.get_pendulum_length_pixels()

        # 2. 事件相机（带圆形 ROI）
        camera_filter = {
            'roi_mode': 'circle',
            'pendulum_length_pixels': px,
            'refractory_us': 500,
        }
        camera_cfg = {
            'dt': 10000,
            'bgnp': 0.001,
            'bgnn': 0.001,
            'th_pos': 0.2,
            'th_neg': 0.2,
        }
        self.event_camera = IntegratedEventCamera(
            self.config['video_width'],
            self.config['video_height'],
            camera_cfg,
            filter_config=camera_filter,
        )

        # 3. 事件直驱控制器（无估计器）
        self.controller = EventDrivenController(
            self.config['video_width'],
            self.config['video_height'],
            {
                'Kp': 60.0,
                'Kd': 120.0,
                'max_force': 10.0,
                'drift_correction_enabled': True,
                'K_cart': 5.0,
                'drift_alpha': 0.005,
                'offset_alpha': 0.25,
            }
        )

        self.running = False
        self.simulation_time = 0.0
        self.frame_count = 0

        # 数据记录
        self.log = {
            't': [], 'angle': [], 'vel': [], 'cart': [],
            'force': [], 'events': [], 'offset': [],
        }

        self.out_dir = "outputs"
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self):
        print("\n--- Starting ---\n")

        dt_physics = 1.0 / self.pendulum.config['sampling_rate']
        dt_main = self.event_camera.config['dt'] * 1e-6
        n_frames = int(self.config['simulation_duration'] / dt_main)
        n_physics = int(dt_main / dt_physics)

        print(f"  Control rate: {1/dt_main:.0f} Hz")
        print(f"  Physics rate: {1/dt_physics:.0f} Hz")
        print(f"  Total frames: {n_frames} ({self.config['simulation_duration']}s)")
        print()

        # 重置
        self.pendulum.reset()
        self.event_camera.reset()
        self.controller.reset()

        frame = self.pendulum.get_current_image()
        self.event_camera.init_with_frame(frame)

        self.running = True
        t_start = time.time()

        for idx in range(n_frames):
            if not self.running:
                break

            self.simulation_time = idx * dt_main

            # 1. 渲染画面
            frame = self.pendulum.get_current_image()

            # 2. 事件相机（带圆形 ROI，跟随小车）
            cart_x = self.pendulum.get_cart_x_pixel()
            events = self.event_camera.process_frame(frame, cart_x_pixel=cart_x)

            # 3. 事件→力（无估计器！）
            force = self.controller.compute_force(events, self.event_camera.current_time_us)

            # 4. 物理步进
            for _ in range(n_physics):
                self.pendulum.step(force)

            # 5. 记录
            angle = self.pendulum.get_angle()
            vel = self.pendulum.get_angular_velocity()
            cart = self.pendulum.get_cart_position()
            self.log['t'].append(self.simulation_time)
            self.log['angle'].append(angle)
            self.log['vel'].append(vel)
            self.log['cart'].append(cart)
            self.log['force'].append(force)
            self.log['events'].append(events.i if events else 0)
            self.log['offset'].append(self.controller.smoothed_offset)

            # 6. 显示
            if self.config['show_display']:
                self._display(frame, force, events.i if events else 0)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.frame_count += 1

        self.running = False
        elapsed = time.time() - t_start

        self._report(elapsed, dt_main)
        cv2.destroyAllWindows()

    def _display(self, frame, force, n_events):
        d = frame.copy()
        a = self.pendulum.get_angle()
        a_deg = np.degrees(a)
        cart = self.pendulum.get_cart_position()

        cv2.putText(d, f"Angle: {a_deg:+5.1f} deg", (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(d, f"Cart: {cart:+5.2f} m", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(d, f"Force: {force:+5.2f} N", (10, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(d, f"Events: {n_events}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        cv2.putText(d, "EVENT->FORCE (no estimator)", (250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # 画面中心线
        cv2.line(d, (self.config['video_width'] // 2, 0),
                 (self.config['video_width'] // 2, 240), (80, 80, 80), 1)

        cv2.imshow("Event-Driven Control", d)

    def _report(self, elapsed, dt_main):
        a = np.degrees(self.log['angle'])
        c = self.log['cart']
        f = self.log['force']

        # 稳态：最后 1/3
        n_ss = max(1, len(a) // 3)
        a_ss = np.array(a[-n_ss:])
        c_ss = np.array(c[-n_ss:])
        f_ss = np.array(f[-n_ss:])

        rms_a = np.sqrt(np.mean(a_ss ** 2))
        max_a = np.max(np.abs(a_ss))
        max_c = np.max(np.abs(c_ss))
        mean_force = np.mean(f_ss)

        # 检查是否振荡：符号变化次数
        sign_changes = np.sum(np.diff(np.sign(f_ss)) != 0)

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  Simulation: {self.simulation_time:.1f}s (wall: {elapsed:.1f}s)")
        print(f"  Final angle: {a[-1]:+.2f} deg")
        print(f"  Final cart:  {c[-1]:+.3f} m")
        print(f"  SS RMS angle: {rms_a:.2f} deg")
        print(f"  SS max angle: {max_a:.2f} deg")
        print(f"  SS max cart:  {max_c:.3f} m")
        print(f"  SS avg force: {mean_force:+.3f} N")
        print(f"  Force sign changes: {sign_changes}")

        # 判断稳定性
        stable = max_a < 10.0
        oscillates = sign_changes > n_ss * 0.1  # 至少 10% 时间在换向
        no_drift = max_c < 2.0

        if stable and oscillates and no_drift:
            print("\n  *** STABLE + OSCILLATING (no drift) ***")
        elif stable:
            print("\n  *** STABLE but check behavior ***")
        else:
            print("\n  *** NOT STABLE ***")

        # 保存数据
        np.savez(f"{self.out_dir}/event_control_result.npz",
                 t=np.array(self.log['t']),
                 angle=a, cart=c, force=f,
                 events=self.log['events'],
                 offset=self.log['offset'])


def main():
    import traceback
    try:
        system = EventDrivenSystem()
        system.run()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
