# closed_loop_system.py
"""
主闭环控制系统
集成：倒立摆模拟器 + 事件相机 + 状态估计 + LQR控制
"""
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 导入自定义模块
from inverted_pendulum_simulator import InvertedPendulumSimulator
from integrated_event_camera import IntegratedEventCamera
from simple_state_estimator import SimpleStateEstimator
from pendulum_controller import PendulumController


class ClosedLoopSystem:
    """闭环控制系统"""

    def __init__(self, config=None):
        """初始化闭环系统"""
        self.config = {
            # 系统参数
            'simulation_duration': 10.0,  # 模拟时长 (秒)
            'real_time_factor': 1.0,  # 实时因子 (1.0 = 实时)

            # 视频参数
            'video_width': 320,
            'video_height': 240,
            'save_video': True,
            'show_display': True,

            # 数据记录
            'log_data': True,
            'log_interval': 0.1,  # 记录间隔 (秒)

            # 性能监控
            'monitor_performance': True,
        }

        if config:
            self.config.update(config)
        # 初始化组件
        print("=" * 60)
        print("初始化闭环控制系统")
        print("=" * 60)

        # 1. 倒立摆模拟器
        pendulum_config = {
            'image_width': self.config['video_width'],
            'image_height': self.config['video_height'],
            'sampling_rate': 100.0,  # 物理模拟频率

        }
        self.pendulum = InvertedPendulumSimulator(pendulum_config)

        # 2. 事件相机模拟器
        event_camera_config = {
            'dt': 10000,  # 事件相机更新间隔 (微秒) = 10ms = 100Hz
        }
        self.event_camera = IntegratedEventCamera(
            self.config['video_width'],
            self.config['video_height'],
            event_camera_config
        )

        # 3. 状态估计器
        estimator_config = {
            'angle_noise_std': 0.05,  # 角度噪声
            'velocity_noise_std': 0.2,  # 角速度噪声
            'delay_frames': 2,
            'use_ground_truth': True,  # 调试时使用真实状态
        }
        self.estimator = SimpleStateEstimator(
            self.config['video_width'],
            self.config['video_height'],
            estimator_config
        )

        # 设置真实状态回调3
        self.estimator.set_ground_truth_callback(
            lambda: (self.pendulum.get_angle(), self.pendulum.get_angular_velocity())
        )

        # 4. 控制器
        controller_config = {
            'controller_type': 'PD',
            'Kp': 80.0,
            'Kd': 15.0,
            'max_force': 8.0,
            'sampling_rate': 100.0,  # 控制器频率
        }
        self.controller = PendulumController(controller_config)

        # 系统状态
        self.running = False
        self.simulation_time = 0.0
        self.frame_count = 0

        # 数据记录
        self.data_log = {
            'timestamps': [],
            'true_angles': [],
            'estimated_angles': [],
            'true_velocities': [],
            'estimated_velocities': [],
            'control_forces': [],
            'event_counts': [],
            'processing_times': [],
        }

        # 性能监控
        self.performance_stats = {
            'avg_loop_time': 0.0,
            'min_loop_time': float('inf'),
            'max_loop_time': 0.0,
            'frame_rate': 0.0,
        }

        # 创建输出目录
        self.output_dir = "outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print("闭环系统初始化完成")
        print(f"输出目录: {self.output_dir}")

    def run_simulation(self):
        """运行闭环模拟"""
        print("\n" + "=" * 60)
        print("开始闭环模拟")
        print("=" * 60)

        # 计算总帧数
        dt_physics = 1.0 / self.pendulum.config['sampling_rate']
        dt_event_camera = self.event_camera.config['dt'] * 1e-6  # 微秒转秒

        # 使用事件相机的时间步长作为主循环步长
        dt_main = dt_event_camera
        total_frames = int(self.config['simulation_duration'] / dt_main)

        print(f"模拟参数:")
        print(f"  总时长: {self.config['simulation_duration']}s")
        print(f"  主循环步长: {dt_main * 1000:.1f}ms")
        print(f"  总帧数: {total_frames}")
        print(f"  实时因子: {self.config['real_time_factor']}")

        # 重置所有组件
        self.pendulum.reset()
        self.event_camera.reset()
        self.estimator.reset()
        self.controller.reset()

        # 初始化事件相机（使用第一帧）
        first_frame = self.pendulum.get_current_image()
        self.event_camera.init_with_frame(first_frame)

        # 主循环
        self.running = True
        start_time = time.time()
        real_start_time = start_time

        # 初始化控制力
        control_force = 0.0

        for frame_idx in range(total_frames):
            if not self.running:
                break

            frame_start_time = time.time()
            self.simulation_time = frame_idx * dt_main

            # 1. 获取当前状态的真实图像
            current_frame = self.pendulum.get_current_image()

            # 2. 事件相机处理（生成事件流）
            events = self.event_camera.process_frame(current_frame)

            # 3. 状态估计（从事件流估计角度和角速度）
            current_time_us = self.event_camera.current_time_us
            estimated_angle, estimated_velocity, valid = self.estimator.estimate_from_events(
                events, current_time_us
            )

            # 4. 控制器计算
            control_force = self.controller.compute_control(
                estimated_angle, estimated_velocity, self.simulation_time
            )

            # 5. 应用控制力到倒立摆（物理模拟）
            # 注意：这里需要多次物理步长来匹配事件相机步长
            physics_steps_per_frame = int(dt_main / dt_physics)
            for _ in range(physics_steps_per_frame):
                self.pendulum.step(control_force)

            # 6. 记录数据
            # if frame_idx % int(1.0 / dt_main / 10) == 0:  # 每秒记录10次
            #     self._log_data(
            #         self.simulation_time,
            #         self.pendulum.get_angle(),
            #         estimated_angle,
            #         self.pendulum.get_angular_velocity(),
            #         estimated_velocity,
            #         control_force,
            #         events.i,
            #         time.time() - frame_start_time
            #     )

            # 7. 显示（可选）
            if self.config['show_display']:
                self._display_current_state(
                    current_frame,
                    self.pendulum.get_angle(),
                    estimated_angle,
                    control_force,
                    frame_idx,
                    events.i
                )

            # 8. 性能监控
            frame_time = time.time() - frame_start_time
            self._update_performance_stats(frame_time)

            # 9. 实时控制
            if self.config['real_time_factor'] > 0:
                # 计算应该等待的时间
                expected_frame_time = dt_main / self.config['real_time_factor']
                if frame_time < expected_frame_time:
                    time.sleep(expected_frame_time - frame_time)

            # 10. 检查退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户请求停止模拟")
                break

        # 模拟结束
        self.running = False
        total_real_time = time.time() - real_start_time

        print("\n" + "=" * 60)
        print("模拟结束")
        print("=" * 60)

        # 显示性能统计
        self._print_performance_stats(total_real_time)

        # 保存数据
        if self.config['log_data']:
            self._save_data()

        # 生成报告
        # self._generate_report()

        # 清理
        cv2.destroyAllWindows()

    def _log_data(self, timestamp, true_angle, est_angle, true_vel, est_vel, force, event_count, proc_time):
        """记录数据"""
        self.data_log['timestamps'].append(timestamp)
        self.data_log['true_angles'].append(true_angle)
        self.data_log['estimated_angles'].append(est_angle)
        self.data_log['true_velocities'].append(true_vel)
        self.data_log['estimated_velocities'].append(est_vel)
        self.data_log['control_forces'].append(force)
        self.data_log['event_counts'].append(event_count)
        self.data_log['processing_times'].append(proc_time)

    def _display_current_state(self, frame, true_angle, est_angle, force, frame_idx, event_count):
        """显示当前状态"""
        display = frame.copy()

        # 添加估计信息
        cv2.putText(display, f"Frame: {frame_idx}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"True Angle: {np.degrees(true_angle):.1f} deg", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display, f"Est Angle: {np.degrees(est_angle):.1f} deg", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(display, f"Force: {force:.2f}N", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"Events: {event_count}", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # 显示系统状态
        status = "RUNNING" if self.running else "STOPPED"
        cv2.putText(display, f"Status: {status}", (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.running else (0, 0, 255), 2)

        cv2.imshow("Closed-Loop Control", display)

    def _update_performance_stats(self, frame_time):
        """更新性能统计"""
        self.performance_stats['avg_loop_time'] = (
                                                          self.performance_stats[
                                                              'avg_loop_time'] * self.frame_count + frame_time
                                                  ) / (self.frame_count + 1)

        self.performance_stats['min_loop_time'] = min(
            self.performance_stats['min_loop_time'], frame_time
        )

        self.performance_stats['max_loop_time'] = max(
            self.performance_stats['max_loop_time'], frame_time
        )

        self.frame_count += 1

        if self.simulation_time > 0:
            self.performance_stats['frame_rate'] = self.frame_count / self.simulation_time

    def _print_performance_stats(self, total_real_time):
        """打印性能统计"""
        print("\n性能统计:")
        print(f"  总模拟时间: {self.simulation_time:.2f}s")
        print(f"  总实际时间: {total_real_time:.2f}s")
        print(f"  实时因子: {self.simulation_time / total_real_time:.2f}")
        print(f"  总帧数: {self.frame_count}")
        print(f"  平均帧率: {self.frame_count / total_real_time:.1f} Hz")
        print(f"  平均循环时间: {self.performance_stats['avg_loop_time'] * 1000:.1f}ms")
        print(f"  最小循环时间: {self.performance_stats['min_loop_time'] * 1000:.1f}ms")
        print(f"  最大循环时间: {self.performance_stats['max_loop_time'] * 1000:.1f}ms")

        # 事件相机统计
        event_stats = self.event_camera.get_event_statistics()
        print(f"\n事件相机统计:")
        print(f"  总事件数: {event_stats['total_events']}")
        print(f"  平均事件率: {event_stats['average_rate']:.0f} events/s")
        print(f"  总帧数: {event_stats['frame_count']}")

        # 控制器统计
        control_stats = self.controller.get_control_statistics()
        print(f"\n控制器统计:")
        print(f"  平均控制力: {control_stats['avg_force']:.2f}N")
        print(f"  最大控制力: {control_stats['max_force']:.2f}N")
        print(f"  RMS角度误差: {np.degrees(control_stats['rms_error']):.2f}°")

#     # def _save_data(self):
#     #     """保存数据到文件"""
#     #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     #
#     #     # 保存原始数据
#     #     data_file = f"{self.output_dir}/simulation_data_{timestamp}.npz"
#     #     np.savez(
#     #         data_file,
#     #         timestamps=np.array(self.data_log['timestamps']),
#     #         true_angles=np.array(self.data_log['true_angles']),
#     #         estimated_angles=np.array(self.data_log['estimated_angles']),
#     #         true_velocities=np.array(self.data_log['true_velocities']),
#     #         estimated_velocities=np.array(self.data_log['estimated_velocities']),
#     #         control_forces=np.array(self.data_log['control_forces']),
#     #         event_counts=np.array(self.data_log['event_counts']),
#     #         processing_times=np.array(self.data_log['processing_times']),
#     #         config=self.config
#     #     )
#     #
#     #     print(f"数据保存到: {data_file}")
#
#     def _generate_report(self):
#         """生成模拟报告和图表"""
#         if not self.data_log['timestamps']:
#             print("没有数据可生成报告")
#             return
#
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#         # 创建图表
#         fig, axes = plt.subplots(3, 2, figsize=(14, 12))
#
#         timestamps = np.array(self.data_log['timestamps'])
#         true_angles = np.degrees(self.data_log['true_angles'])
#         est_angles = np.degrees(self.data_log['estimated_angles'])
#         control_forces = self.data_log['control_forces']
#         event_counts = self.data_log['event_counts']
#
#         # 1. 角度对比
#         axes[0, 0].plot(timestamps, true_angles, 'b-', label='真实角度', linewidth=2)
#         axes[0, 0].plot(timestamps, est_angles, 'r--', label='估计角度', linewidth=2, alpha=0.7)
#         axes[0, 0].set_xlabel('时间 (s)')
#         axes[0, 0].set_ylabel('角度 (°)')
#         axes[0, 0].set_title('角度对比')
#         axes[0, 0].legend()
#         axes[0, 0].grid(True, alpha=0.3)
#
#         # 2. 角度误差
#         angle_errors = np.abs(np.array(true_angles) - np.array(est_angles))
#         axes[0, 1].plot(timestamps, angle_errors, 'g-', linewidth=1)
#         axes[0, 1].set_xlabel('时间 (s)')
#         axes[0, 1].set_ylabel('角度误差 (°)')
#         axes[0, 1].set_title(f'角度误差 (平均: {np.mean(angle_errors):.2f}°)')
#         axes[0, 1].grid(True, alpha=0.3)
#
#         # 3. 控制力
#         axes[1, 0].plot(timestamps, control_forces, 'm-', linewidth=2)
#         axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
#         axes[1, 0].set_xlabel('时间 (s)')
#         axes[1, 0].set_ylabel('控制力 (N)')
#         axes[1, 0].set_title('控制力历史')
#         axes[1, 0].grid(True, alpha=0.3)
#
#         # 4. 事件计数
#         axes[1, 1].plot(timestamps, event_counts, 'c-', linewidth=1)
#         axes[1, 1].set_xlabel('时间 (s)')
#         axes[1, 1].set_ylabel('事件数')
#         axes[1, 1].set_title(f'事件计数 (平均: {np.mean(event_counts):.0f})')
#         axes[1, 1].grid(True, alpha=0.3)
#
#         # 5. 相位图（角度 vs 角速度）
#         if len(self.data_log['true_velocities']) > 0:
#             true_velocities = np.degrees(self.data_log['true_velocities'])
#             axes[2, 0].plot(true_angles, true_velocities, 'b-', alpha=0.5)
#             axes[2, 0].set_xlabel('角度 (°)')
#             axes[2, 0].set_ylabel('角速度 (°/s)')
#             axes[2, 0].set_title('相位图 (真实状态)')
#             axes[2, 0].grid(True, alpha=0.3)
#             axes[2, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
#             axes[2, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
#
#         # 6. 控制力直方图
#         axes[2, 1].hist(control_forces, bins=30, alpha=0.7, edgecolor='black')
#         axes[2, 1].axvline(x=np.mean(control_forces), color='r', linestyle='--',
#                            label=f'均值: {np.mean(control_forces):.2f}N')
#         axes[2, 1].set_xlabel('控制力 (N)')
#         axes[2, 1].set_ylabel('频次')
#         axes[2, 1].set_title('控制力分布')
#         axes[2, 1].legend()
#         axes[2, 1].grid(True, alpha=0.3)
#
#         plt.tight_layout()
#         report_file = f"{self.output_dir}/simulation_report_{timestamp}.png"
#         plt.savefig(report_file, dpi=150)
#         plt.show()
#
#         print(f"报告保存到: {report_file}")
#
#         # 生成文本报告
#         text_report = f"""闭环控制系统模拟报告
# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#
# 模拟参数:
#   模拟时长: {self.config['simulation_duration']}s
#   实时因子: {self.config['real_time_factor']}
#   视频分辨率: {self.config['video_width']}x{self.config['video_height']}
#
# 性能统计:
#   总帧数: {self.frame_count}
#   平均帧率: {self.performance_stats['frame_rate']:.1f} Hz
#   实时因子: {self.simulation_time / (time.time() - self.performance_stats.get('start_time', time.time())):.2f}
#
# 控制性能:
#   平均角度误差: {np.mean(angle_errors):.2f}°
#   RMS角度误差: {np.sqrt(np.mean(angle_errors ** 2)):.2f}°
#   平均控制力: {np.mean(np.abs(control_forces)):.2f}N
#   最大控制力: {np.max(np.abs(control_forces)):.2f}N
#
# 事件相机统计:
#   总事件数: {np.sum(event_counts)}
#   平均事件率: {np.mean(event_counts) / (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0:.0f} events/s
# """
#
#         report_text_file = f"{self.output_dir}/simulation_report_{timestamp}.txt"
#         with open(report_text_file, 'w') as f:
#             f.write(text_report)
#
#         print(f"文本报告保存到: {report_text_file}")
#         print("\n" + text_report)


def main():
    """主函数"""
    print("倒立摆闭环控制系统模拟器")
    print("=" * 60)

    # 配置
    config = {
        'simulation_duration': 20.0,  # 模拟5秒
        'real_time_factor': 1.0,  # 实时运行
        'video_width': 480,
        'video_height': 360,
        'save_video': False,
        'show_display': True,
        'log_data': True,
    }

    # 创建系统
    system = ClosedLoopSystem(config)

    # 运行模拟
    try:
        system.run_simulation()
    except KeyboardInterrupt:
        print("\n模拟被用户中断")
    except Exception as e:
        print(f"\n模拟出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("\n系统关闭完成")


if __name__ == "__main__":
    main()