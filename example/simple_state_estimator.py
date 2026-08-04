# simple_state_estimator.py
"""
Simplified state estimator using ground truth with added noise.

输出全部 4 个状态：小车位置、小车速度、角度、角速度。
"""
import numpy as np


class SimpleStateEstimator:
    def __init__(self, width, height, config=None):
        self.width = width
        self.height = height
        self.config = {
            'use_ground_truth': True,
            'angle_noise_std': 0.005,   # rad
            'vel_noise_std': 0.05,      # rad/s
            'cart_noise_std': 0.01,     # m
            'cart_vel_noise_std': 0.05, # m/s
        }
        if config:
            self.config.update(config)
        print(f"SimpleStateEstimator initialized: 4-state ground truth")

    def set_ground_truth_callback(self, callback):
        """回调返回 (angle, ang_vel, cart_pos, cart_vel)"""
        self.ground_truth_callback = callback

    def estimate_from_events(self, events, current_time_us):
        """返回 4 个状态：angle, ang_vel, valid, cart_pos, cart_vel"""
        if self.config['use_ground_truth'] and hasattr(self, 'ground_truth_callback'):
            result = self.ground_truth_callback()
            # 兼容 2-state 或 4-state 的回调
            if len(result) == 2:
                angle, ang_vel = result
                cart_pos, cart_vel = 0.0, 0.0
            else:
                angle, ang_vel, cart_pos, cart_vel = result

            angle += np.random.normal(0, self.config['angle_noise_std'])
            ang_vel += np.random.normal(0, self.config['vel_noise_std'])
            cart_pos += np.random.normal(0, self.config['cart_noise_std'])
            cart_vel += np.random.normal(0, self.config['cart_vel_noise_std'])
            valid = True
        else:
            angle = 0.0
            ang_vel = 0.0
            cart_pos = 0.0
            cart_vel = 0.0
            valid = False

        return angle, ang_vel, valid, cart_pos, cart_vel

    def reset(self):
        pass
