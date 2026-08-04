# event_controller.py
"""
Event-driven direct controller for inverted pendulum.

Maps event camera signals directly to control force WITHOUT
going through any state estimator.

Features extracted from events:
  1. Event centroid x-offset (normalized by width) → 摆角 + 小车位置
  2. Left-right event count imbalance → 摆角方向
  3. Weighted centroid velocity → 阻尼

Control law: F = -(Kp * offset + Kd * velocity)
               + K_cart * cart_bias_correction

负号含义：事件质心偏右 → 需要向左推 → F < 0
"""
import numpy as np


class EventDrivenController:
    def __init__(self, width, height, config=None):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.ground_y = int(height * 2 / 3)

        self.config = {
            # 事件滤波参数
            'x_margin': 10,
            'y_min': 5,
            'y_max': None,  # 自动设
            'min_events': 3,

            # PD 增益（直接事件→力）
            'Kp': 60.0,      # 比例——主要稳定杆子
            'Kd': 120.0,     # 微分——阻尼
            'max_force': 10.0,

            # 平滑
            'offset_alpha': 0.25,    # 质心 EMA
            'velocity_alpha': 0.12,  # 速度 EMA

            # 防漂移：当长期平均质心偏离中心时，缓慢修正
            'drift_correction_enabled': True,
            'K_cart': 5.0,          # 漂移修正增益
            'drift_alpha': 0.005,    # 漂移估计的 EMA 系数（非常慢）
            'drift_max': 0.5,        # 最大归一化漂移修正

            # 事件率自适应增益
            'rate_gain_min': 0.5,
            'rate_gain_max': 1.5,
            'rate_low': 20,
            'rate_high': 300,
        }

        if config:
            self.config.update(config)
        if self.config['y_max'] is None:
            self.config['y_max'] = self.ground_y - 25

        # 连续状态
        self.smoothed_offset = 0.0
        self.smoothed_velocity = 0.0
        self.drift_offset = 0.0        # 慢变漂移估计
        self.prev_centroid = float(self.center_x)
        self.prev_time_us = 0

        # 仅用于调试记录
        self.log = {'offset': [], 'vel': [], 'force': [], 'drift': []}

        print(f"EventDrivenController (direct event-to-force)")
        print(f"  F = -({self.config['Kp']}*offset + {self.config['Kd']}*vel) + drift_correction")
        print(f"  drift_correction enabled={self.config['drift_correction_enabled']}")

    def compute_force(self, events, current_time_us=None):
        """将事件直接映射为控制力。"""
        offset, velocity = self._extract_features(events, current_time_us)

        # EMA 平滑
        oa = self.config['offset_alpha']
        self.smoothed_offset = oa * offset + (1 - oa) * self.smoothed_offset

        va = self.config['velocity_alpha']
        self.smoothed_velocity = va * velocity + (1 - va) * self.smoothed_velocity

        # 事件率自适应增益
        n = events.i if events else 0
        rate_gain = np.clip(
            (n - self.config['rate_low']) / (self.config['rate_high'] - self.config['rate_low']),
            self.config['rate_gain_min'], self.config['rate_gain_max']
        )

        # 漂移修正：缓慢跟踪质心的长期平均偏移
        if self.config['drift_correction_enabled']:
            da = self.config['drift_alpha']
            self.drift_offset = (1 - da) * self.drift_offset + da * self.smoothed_offset
            drift_correction = np.clip(
                self.config['K_cart'] * self.drift_offset,
                -self.config['drift_max'], self.config['drift_max']
            )
        else:
            drift_correction = 0.0

        # 控制律
        Kp = self.config['Kp'] * rate_gain
        Kd = self.config['Kd'] * rate_gain
        F = -(Kp * self.smoothed_offset + Kd * self.smoothed_velocity) + drift_correction
        F = float(np.clip(F, -self.config['max_force'], self.config['max_force']))

        # 记录
        if len(self.log['force']) < 10000:
            self.log['offset'].append(self.smoothed_offset)
            self.log['vel'].append(self.smoothed_velocity)
            self.log['force'].append(F)
            self.log['drift'].append(drift_correction)

        return F

    def _extract_features(self, events, current_time_us):
        """从事件中提取归一化质心偏移和质心速度。"""
        if events is None or events.i < self.config['min_events']:
            self.smoothed_velocity *= 0.95
            return self.smoothed_offset, self.smoothed_velocity

        x_all = events.get_x()
        y_all = events.get_y()
        p_all = events.get_p()

        # ROI 滤波
        m = self.config['x_margin']
        ym = self.config['y_min']
        yx = self.config['y_max']
        mask = (x_all > m) & (x_all < self.width - m) & (y_all > ym) & (y_all < yx)
        x, y, p = x_all[mask], y_all[mask], p_all[mask]
        n = len(x)

        if n < self.config['min_events']:
            self.smoothed_velocity *= 0.95
            return self.smoothed_offset, self.smoothed_velocity

        # 只使用 ON 事件 (p==1)，它们反映杆子当前位置
        on = p == 1
        if np.any(on):
            x_on, y_on = x[on], y[on]
        else:
            x_on, y_on = x, y

        # 加权质心：上方像素（更靠近杆子顶端）权重更大
        y_max_f = float(yx)
        w = (y_max_f - y_on.astype(float)) ** 2 + 1.0
        centroid = np.average(x_on, weights=w)

        # 归一化偏移 [-1, 1]
        offset = 2.0 * (centroid - self.center_x) / self.width

        # 质心速度
        velocity = 0.0
        if current_time_us is not None and self.prev_time_us > 0:
            dt = (current_time_us - self.prev_time_us) * 1e-6
            if dt > 1e-6:
                v_raw = (centroid - self.prev_centroid) / dt
                velocity = v_raw / (self.width // 2)

        self.prev_centroid = centroid
        if current_time_us is not None:
            self.prev_time_us = current_time_us

        return offset, velocity

    def reset(self):
        self.smoothed_offset = 0.0
        self.smoothed_velocity = 0.0
        self.drift_offset = 0.0
        self.prev_centroid = float(self.center_x)
        self.prev_time_us = 0
        self.log = {'offset': [], 'vel': [], 'force': [], 'drift': []}
