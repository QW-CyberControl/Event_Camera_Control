# pendulum_controller.py
"""
Inverted pendulum controller module

支持多种控制策略：
  - PD:         仅角度 + 角速度
  - PID:        角度 + 角速度 + 积分
  - LQR:        全状态反馈 (cart_pos + cart_vel + angle + angular_vel)
  - BangBang:   开关控制（仅角度）

LQR 控制律:
  F = -(K_cart_pos * cart_pos + K_cart_vel * cart_vel
        + K_angle * angle + K_angle_vel * angular_vel)
"""
import numpy as np


class PendulumController:
    """Inverted pendulum controller"""

    def __init__(self, config=None):
        self.config = {
            'controller_type': 'LQR',  # PD, PID, LQR, BangBang

            # LQR 全状态增益
            'K_cart_pos': 2.0,      # 小车位置增益（防漂移）
            'K_cart_vel': 3.0,      # 小车速度阻尼
            'K_angle': 80.0,        # 角度增益（稳定杆子）
            'K_angle_vel': 15.0,    # 角速度阻尼

            # PD 增益（向后兼容）
            'Kp': 50.0,
            'Kd': 10.0,
            'Ki': 0.0,
            'integral_limit': 5.0,

            'max_force': 8.0,       # 最大控制力
            'target_angle': 0.0,    # 目标角度（竖直向上）
            'target_cart': 0.0,     # 目标小车位置（中心）
            'deadband': 0.005,      # 死区

            'sampling_rate': 100.0,
        }

        if config:
            self.config.update(config)

        # 状态
        self.integral_error = 0.0
        self.prev_error = 0.0

        # 历史
        self.control_history = []
        self.error_history = []
        self.cart_error_history = []

        print(f"Controller initialized: {self.config['controller_type']} controller")
        if self.config['controller_type'] == 'LQR':
            print(f"  Gains: Kx={self.config['K_cart_pos']}, Kv={self.config['K_cart_vel']}, "
                  f"Kth={self.config['K_angle']}, Kw={self.config['K_angle_vel']}")
        else:
            print(f"  Gains: Kp={self.config['Kp']}, Kd={self.config['Kd']}, Ki={self.config['Ki']}")

    def compute_control(self, angle, angular_velocity, current_time=None,
                        cart_pos=0.0, cart_velocity=0.0):
        """
        Compute control force.

        LQR 控制律（标准形式）:
            u = -K * [x, x_dot, theta, theta_dot]
          = -(K_cart_pos * cart_pos + K_cart_vel * cart_vel
              + K_angle * angle + K_angle_vel * angular_vel)

        注意：因为目标状态是零（杆子竖直、小车居中），
        状态本身就是误差信号，不需要再算 error = target - state。

        Args:
            angle: 当前角度 (rad)
            angular_velocity: 当前角速度 (rad/s)
            current_time: 当前时间 (s)
            cart_pos: 小车位置 (m)
            cart_velocity: 小车速度 (m/s)

        Returns:
            control_force (N)
        """
        # 角度误差（兼容 PD/PID）
        angle_error = self.config['target_angle'] - angle
        if abs(angle_error) < self.config['deadband']:
            angle_error = 0.0

        # 积分项（兼容 PID）
        self.integral_error += angle_error / self.config['sampling_rate']
        self.integral_error = np.clip(
            self.integral_error,
            -self.config['integral_limit'],
            self.config['integral_limit']
        )

        # 按控制器类型计算力
        if self.config['controller_type'] == 'LQR':
            # LQR: u = -K * state
            # state = [cart_pos, cart_vel, angle, ang_vel]
            control_force = -(
                self.config['K_cart_pos'] * cart_pos +
                self.config['K_cart_vel'] * cart_velocity +
                self.config['K_angle'] * angle +
                self.config['K_angle_vel'] * angular_velocity
            )

        elif self.config['controller_type'] == 'PD':
            control_force = (
                self.config['Kp'] * angle_error +
                self.config['Kd'] * (-angular_velocity)
            )

        elif self.config['controller_type'] == 'PID':
            control_force = (
                self.config['Kp'] * angle_error +
                self.config['Kd'] * (-angular_velocity) +
                self.config['Ki'] * self.integral_error
            )

        elif self.config['controller_type'] == 'BangBang':
            if angle_error > 0:
                control_force = self.config['max_force']
            else:
                control_force = -self.config['max_force']

        else:
            control_force = (
                self.config['Kp'] * angle_error +
                self.config['Kd'] * (-angular_velocity)
            )

        # 限幅
        control_force = np.clip(
            control_force,
            -self.config['max_force'],
            self.config['max_force']
        )

        # 记录
        self.control_history.append(control_force)
        self.error_history.append(angle_error)
        # cart_error = target_cart - cart_pos（仅用于记录）
        _cart_error = self.config['target_cart'] - cart_pos
        self.cart_error_history.append(_cart_error)

        return control_force

    def reset(self):
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.control_history = []
        self.error_history = []
        self.cart_error_history = []

    def get_control_statistics(self):
        if not self.control_history:
            return {
                'avg_force': 0.0, 'max_force': 0.0,
                'rms_error': 0.0, 'rms_cart_error': 0.0,
                'num_control_steps': 0,
            }

        avg_force = np.mean(np.abs(self.control_history))
        max_force = np.max(np.abs(self.control_history))

        rms_error = 0.0
        if self.error_history:
            rms_error = np.sqrt(np.mean(np.array(self.error_history) ** 2))

        rms_cart = 0.0
        if self.cart_error_history:
            rms_cart = np.sqrt(np.mean(np.array(self.cart_error_history) ** 2))

        return {
            'avg_force': avg_force,
            'max_force': max_force,
            'rms_error': rms_error,
            'rms_cart_error': rms_cart,
            'num_control_steps': len(self.control_history),
        }
