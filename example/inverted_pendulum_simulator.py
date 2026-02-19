# pendulum_simulator.py
"""
倒立摆物理模拟器
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class InvertedPendulumSimulator:
    """倒立摆物理模拟器"""

    def __init__(self, config=None):
        """初始化倒立摆模拟器"""
        # 物理参数
        self.config = {
            'length': 1.0,  # 摆杆长度 (m)
            'mass': 0.1,  # 摆杆质量 (kg)
            'gravity': 9.81,  # 重力加速度 (m/s²)
            'friction': 0.1,  # 摩擦系数
            'cart_mass': 1.0,  # 小车质量 (kg)
            'cart_friction': 0.05,  # 小车摩擦系数

            # 图像参数
            'image_width': 320,
            'image_height': 240,
            'pendulum_thickness': 5,
            'cart_width': 40,
            'cart_height': 20,

            # 控制参数
            'max_force': 10.0,  # 最大控制力 (N)
            'sampling_rate': 100.0,  # 采样率 (Hz)

            # 初始状态
            'initial_angle': np.radians(5),  # 初始角度 (rad)
            'initial_angular_velocity': 0.0,  # 初始角速度 (rad/s)
            'initial_cart_position': 0.0,  # 初始小车位置 (m)
            'initial_cart_velocity': 0.0,  # 初始小车速度 (m/s)
        }

        if config:
            self.config.update(config)

        # 状态变量
        self.state = np.array([
            self.config['initial_cart_position'],
            self.config['initial_cart_velocity'],
            self.config['initial_angle'],
            self.config['initial_angular_velocity']
        ])

        self.time = 0.0
        self.dt = 1.0 / self.config['sampling_rate']

        # 控制输入历史
        self.control_history = []
        self.state_history = []
        self.time_history = []

        # 图像帧缓存
        self.frame_buffer = []

        # 预计算参数
        self.m = self.config['mass']
        self.M = self.config['cart_mass']
        self.l = self.config['length']
        self.g = self.config['gravity']
        self.b = self.config['friction']
        self.b_cart = self.config['cart_friction']

        print(f"倒立摆模拟器初始化完成:")
        print(f"  摆杆长度: {self.l}m, 质量: {self.m}kg")
        print(f"  小车质量: {self.M}kg")
        print(f"  采样率: {self.config['sampling_rate']}Hz")
        print(f"  初始角度: {np.degrees(self.state[2]):.1f}°")

    def dynamics(self, t, state, F):
        """
        倒立摆动力学方程

        Args:
            t: 时间
            state: [x, x_dot, theta, theta_dot]
            F: 控制力 (N)

        Returns:
            状态导数
        """
        x, x_dot, theta, theta_dot = state

        # 系统参数
        m, M, l, g, b, b_cart = self.m, self.M, self.l, self.g, self.b, self.b_cart

        # 中间变量
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # 计算加速度
        denominator = (M + m) * (l ** 2) - m ** 2 * l ** 2 * cos_theta ** 2

        if abs(denominator) < 1e-6:
            denominator = 1e-6

        # 角度加速度
        theta_ddot = (
                             (M + m) * g * sin_theta
                             - m * l * theta_dot ** 2 * sin_theta * cos_theta
                             + (F - b_cart * x_dot) * cos_theta
                             - b * theta_dot
                     ) / denominator * l

        # 小车加速度
        x_ddot = (
                         F - b_cart * x_dot
                         + m * l * (theta_dot ** 2 * sin_theta - theta_ddot * cos_theta)
                 ) / (M + m)

        return [x_dot, x_ddot, theta_dot, theta_ddot]

    def step(self, control_force=0.0):
        """
        执行一个时间步长的模拟

        Args:
            control_force: 控制力 (N)

        Returns:
            更新后的状态
        """
        # 限制控制力
        control_force = np.clip(
            control_force,
            -self.config['max_force'],
            self.config['max_force']
        )

        # 使用RK4积分
        k1 = self.dynamics(self.time, self.state, control_force)
        k2 = self.dynamics(self.time + self.dt / 2,
                           self.state + np.array(k1) * self.dt / 2,
                           control_force)
        k3 = self.dynamics(self.time + self.dt / 2,
                           self.state + np.array(k2) * self.dt / 2,
                           control_force)
        k4 = self.dynamics(self.time + self.dt,
                           self.state + np.array(k3) * self.dt,
                           control_force)

        # 更新状态
        self.state += (np.array(k1) + 2 * np.array(k2) + 2 * np.array(k3) + np.array(k4)) * self.dt / 6

        # 更新时间
        self.time += self.dt

        # 记录历史
        self.state_history.append(self.state.copy())
        self.control_history.append(control_force)
        self.time_history.append(self.time)

        return self.state.copy()

    def get_current_image(self):
        """
        生成当前状态的图像

        Returns:
            当前状态的BGR图像
        """
        width = self.config['image_width']
        height = self.config['image_height']

        # 创建空白图像
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # 计算摆杆端点位置
        x_cart = self.state[0]  # 小车位置
        theta = self.state[2]  # 摆杆角度

        # 将物理坐标转换为图像坐标
        # 图像中心对应x=0，底部对应y=0
        ground_y = int(height * 2 / 3)
        vertical_scale = (ground_y - 20) / (2 * self.l)
        # 水平方向：左右各留1/4宽度，摆杆摆动不超出左右边界
        horizontal_scale = width / (4 * self.l)
        # 取较小值确保双向不超出
        scale = min(horizontal_scale, vertical_scale) # 缩放因子
        # scale = min(width, height) / (3 * self.l)

        center_x = width // 2


        # 小车位置
        cart_x = int(center_x + x_cart * scale)
        cart_y = ground_y

        # 摆杆端点
        pendulum_length_pixels = int(self.l * scale)
        pendulum_end_x = int(cart_x + pendulum_length_pixels * np.sin(theta))
        pendulum_end_y = int(cart_y - pendulum_length_pixels * np.cos(theta))

        # 绘制地面
        cv2.line(image, (0, ground_y), (width, ground_y), (200, 200, 200), 2)

        # 绘制小车
        cart_width = self.config['cart_width']
        cart_height = self.config['cart_height']
        cv2.rectangle(image,
                      (cart_x - cart_width // 2, cart_y - cart_height),
                      (cart_x + cart_width // 2, cart_y),
                      (100, 100, 255), -1)

        # 绘制摆杆
        pendulum_thickness = self.config['pendulum_thickness']
        cv2.line(image,
                 (cart_x, cart_y - cart_height // 2),
                 (pendulum_end_x, pendulum_end_y),
                 (0, 255, 0), pendulum_thickness)

        # 绘制摆锤
        cv2.circle(image, (pendulum_end_x, pendulum_end_y),
                   pendulum_thickness * 2, (0, 200, 0), -1)

        # 添加文字信息
        cv2.putText(image, f"Time: {self.time:.2f}s", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Angle: {np.degrees(theta):.1f} deg", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Control: {self.control_history[-1] if self.control_history else 0:.2f}N",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Cart Pos: {x_cart:.2f}m", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image

    def generate_video_frames(self, duration, control_forces=None, save_video=False):
        """
        生成视频帧

        Args:
            duration: 模拟时长 (秒)
            control_forces: 控制力序列 (可选)
            save_video: 是否保存为视频文件

        Returns:
            图像帧列表
        """
        num_steps = int(duration * self.config['sampling_rate'])
        frames = []

        print(f"生成 {duration}s 的视频，共 {num_steps} 帧")

        for i in range(num_steps):
            # 获取控制力
            if control_forces is not None and i < len(control_forces):
                control_force = control_forces[i]
            else:
                control_force = 0.0

            # 模拟一步
            self.step(control_force)

            # 生成图像
            frame = self.get_current_image()
            frames.append(frame)

            # 显示进度
            if i % 100 == 0:
                print(f"  进度: {i}/{num_steps} 帧")

        # 保存视频
        if save_video and frames:
            self.save_video(frames, "outputs/pendulum_simulation.mp4")

        return frames

    def save_video(self, frames, filename):
        """保存视频文件"""
        if not frames:
            return

        height, width = frames[0].shape[:2]
        fps = self.config['sampling_rate']

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
        print(f"视频保存到: {filename}")

    def reset(self):
        """重置模拟器"""
        self.state = np.array([
            self.config['initial_cart_position'],
            self.config['initial_cart_velocity'],
            self.config['initial_angle'],
            self.config['initial_angular_velocity']
        ])
        self.time = 0.0
        self.control_history = []
        self.state_history = []
        self.time_history = []
        self.frame_buffer = []

        print("模拟器已重置")

    def get_state_vector(self):
        """获取状态向量"""
        return self.state.copy()

    def get_angle(self):
        """获取当前角度"""
        return self.state[2]

    def get_angular_velocity(self):
        """获取当前角速度"""
        return self.state[3]

    def get_cart_position(self):
        """获取小车位置"""
        return self.state[0]