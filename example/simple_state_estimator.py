# simple_state_estimator.py
"""
简化的状态估计器，用于闭环控制
"""
import numpy as np
import cv2
from collections import deque


class SimpleStateEstimator:
    """
    简化的状态估计器

    注意：为了闭环测试，我们可以使用不准确但快速的估计
    """

    def __init__(self, width, height, config=None):
        self.width = width
        self.height = height

        # 配置
        self.config = {
            'accumulation_time_us': 20000,
            'history_size': 10,
            'angle_noise_std': 0.1,  # 角度噪声标准差 (rad)
            'velocity_noise_std': 0.5,  # 角速度噪声标准差 (rad/s)
            'delay_frames': 2,  # 模拟延迟
            'use_ground_truth': False,  # 如果为True，使用真实状态（调试用）
        }

        if config:
            self.config.update(config)

        # 状态历史
        self.angle_history = deque(maxlen=self.config['history_size'])
        self.velocity_history = deque(maxlen=self.config['history_size'])
        self.time_history = deque(maxlen=self.config['history_size'])

        # 延迟缓冲区
        self.delay_buffer = deque(maxlen=self.config['delay_frames'])

        # 事件累积图像
        self.event_accumulator = np.zeros((height, width), dtype=np.float32)

        # 状态
        self.current_angle = 0.0
        self.current_velocity = 0.0

        print(f"简化状态估计器初始化完成")

    def estimate_from_events(self, events, current_time_us):
        """
        从事件流估计状态

        Args:
            events: 事件缓冲区
            current_time_us: 当前时间 (微秒)

        Returns:
            angle, angular_velocity, valid
        """
        # 简化估计：假设我们知道真实状态，添加噪声和延迟
        # 在实际系统中，这里应该从事件流中提取状态

        # 方法1：如果允许使用真实状态（调试）
        if self.config['use_ground_truth'] and hasattr(self, 'ground_truth_callback'):
            true_angle, true_velocity = self.ground_truth_callback()

            # 添加噪声
            angle = true_angle + np.random.normal(0, self.config['angle_noise_std'])
            velocity = true_velocity + np.random.normal(0, self.config['velocity_noise_std'])

            valid = True

        else:
            # 方法2：简单的事件处理
            # 这里只是示例，实际需要实现事件到角度的转换
            if events.i > 0:
                # 简单的事件计数方法
                event_density = events.i / (self.width * self.height)

                # 假设事件密度与角度变化率相关
                angle_change = event_density * 0.1  # 比例因子

                # 更新角度
                self.current_angle += angle_change

                # 限制角度范围
                if self.current_angle > np.pi:
                    self.current_angle -= 2 * np.pi
                elif self.current_angle < -np.pi:
                    self.current_angle += 2 * np.pi

                # 计算角速度（简化）
                if len(self.angle_history) > 0:
                    dt = current_time_us * 1e-6 - self.time_history[-1]
                    if dt > 0:
                        self.current_velocity = (self.current_angle - self.angle_history[-1]) / dt
                    else:
                        self.current_velocity = 0.0
                else:
                    self.current_velocity = 0.0

                valid = True
            else:
                # 没有事件，保持当前估计
                valid = False

        # 添加到历史
        self.angle_history.append(self.current_angle)
        self.velocity_history.append(self.current_velocity)
        self.time_history.append(current_time_us * 1e-6)

        # 添加延迟
        self.delay_buffer.append((self.current_angle, self.current_velocity))

        # 使用延迟后的状态
        if len(self.delay_buffer) >= self.config['delay_frames']:
            delayed_angle, delayed_velocity = self.delay_buffer[0]
        else:
            delayed_angle, delayed_velocity = self.current_angle, self.current_velocity

        return delayed_angle, delayed_velocity, valid

    def set_ground_truth_callback(self, callback):
        """设置真实状态回调函数（用于调试）"""
        self.ground_truth_callback = callback

    def reset(self):
        """重置估计器"""
        self.angle_history.clear()
        self.velocity_history.clear()
        self.time_history.clear()
        self.delay_buffer.clear()
        self.event_accumulator.fill(0)
        self.current_angle = 0.0
        self.current_velocity = 0.0

        print("状态估计器已重置")