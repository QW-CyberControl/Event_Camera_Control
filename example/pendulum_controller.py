# pendulum_controller.py
"""
倒立摆控制器模块
"""
import numpy as np


class PendulumController:
    """倒立摆控制器"""

    def __init__(self, config=None):
        self.config = {
            'controller_type': 'PD',  # 控制器类型: PD, LQR, BangBang
            'Kp': 50.0,  # 比例增益
            'Kd': 10.0,  # 微分增益
            'Ki': 0.0,  # 积分增益
            'max_force': 10.0,  # 最大控制力 (N)
            'target_angle': 0.0,  # 目标角度 (rad) - 直立
            'integral_limit': 5.0,  # 积分项限制
            'deadband': 0.01,  # 死区 (rad)
            'sampling_rate': 100.0,  # 采样率 (Hz)
        }

        if config:
            self.config.update(config)

        # 控制器状态
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_time = 0.0

        # 控制历史
        self.control_history = []
        self.error_history = []

        print(f"控制器初始化完成: {self.config['controller_type']}控制器")
        print(f"  增益: Kp={self.config['Kp']}, Kd={self.config['Kd']}, Ki={self.config['Ki']}")

    def compute_control(self, angle, angular_velocity, current_time=None):
        """
        计算控制力

        Args:
            angle: 当前角度估计 (rad)
            angular_velocity: 当前角速度估计 (rad/s)
            current_time: 当前时间 (s)，用于计算微分

        Returns:
            控制力 (N)
        """
        # 计算角度误差（目标角度为0，直立状态）
        error = self.config['target_angle'] - angle

        # 死区处理
        if abs(error) < self.config['deadband']:
            error = 0.0

        # 更新积分项
        self.integral_error += error / self.config['sampling_rate']

        # 积分项限制
        self.integral_error = np.clip(
            self.integral_error,
            -self.config['integral_limit'],
            self.config['integral_limit']
        )

        # 根据控制器类型计算控制力
        if self.config['controller_type'] == 'PD':
            # PD控制器
            control_force = (
                    self.config['Kp'] * error +
                    self.config['Kd'] * (-angular_velocity)  # 负号因为要阻尼
            )

        elif self.config['controller_type'] == 'PID':
            # PID控制器
            control_force = (
                    self.config['Kp'] * error +
                    self.config['Kd'] * (-angular_velocity) +
                    self.config['Ki'] * self.integral_error
            )

        elif self.config['controller_type'] == 'LQR':
            # 简化的LQR控制器（需要状态反馈）
            # 这里使用PD近似
            control_force = (
                    self.config['Kp'] * error +
                    self.config['Kd'] * (-angular_velocity)
            )

        elif self.config['controller_type'] == 'BangBang':
            # Bang-Bang控制器
            if error > 0:
                control_force = self.config['max_force']
            else:
                control_force = -self.config['max_force']

        else:
            # 默认PD控制器
            control_force = (
                    self.config['Kp'] * error +
                    self.config['Kd'] * (-angular_velocity)
            )

        # 限制控制力
        control_force = np.clip(
            control_force,
            -self.config['max_force'],
            self.config['max_force']
        )

        # 记录历史
        self.control_history.append(control_force)
        self.error_history.append(error)

        return control_force

    def reset(self):
        """重置控制器"""
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.control_history = []
        self.error_history = []

        print("控制器已重置")

    def get_control_statistics(self):
        """获取控制统计信息"""
        if not self.control_history:
            return {
                'avg_force': 0.0,
                'max_force': 0.0,
                'rms_error': 0.0
            }

        avg_force = np.mean(np.abs(self.control_history))
        max_force = np.max(np.abs(self.control_history))

        if self.error_history:
            rms_error = np.sqrt(np.mean(np.array(self.error_history) ** 2))
        else:
            rms_error = 0.0

        return {
            'avg_force': avg_force,
            'max_force': max_force,
            'rms_error': rms_error,
            'num_control_steps': len(self.control_history)
        }