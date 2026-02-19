# realtime_monitor.py
"""
实时监控和调试界面
"""
import numpy as np
import cv2
import threading
import time
from queue import Queue


class RealtimeMonitor:
    """实时监控界面"""

    def __init__(self, closed_loop_system):
        self.system = closed_loop_system
        self.data_queue = Queue()
        self.running = False
        self.monitor_thread = None

        # 显示窗口
        self.window_name = "实时监控"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # 监控数据
        self.monitor_data = {
            'angle_history': [],
            'control_history': [],
            'event_rate_history': [],
            'timestamps': [],
        }

        print("实时监控器初始化完成")

    def start(self):
        """启动监控"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

        print("实时监控已启动")

    def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()

        cv2.destroyAllWindows()
        print("实时监控已停止")

    def update_data(self, timestamp, angle, control_force, event_count):
        """更新监控数据"""
        self.data_queue.put({
            'timestamp': timestamp,
            'angle': angle,
            'control_force': control_force,
            'event_count': event_count,
        })

    def _monitor_loop(self):
        """监控主循环"""
        history_size = 100

        while self.running:
            # 获取最新数据
            latest_data = None
            while not self.data_queue.empty():
                latest_data = self.data_queue.get()

            if latest_data:
                # 更新历史数据
                self.monitor_data['timestamps'].append(latest_data['timestamp'])
                self.monitor_data['angle_history'].append(latest_data['angle'])
                self.monitor_data['control_history'].append(latest_data['control_force'])
                self.monitor_data['event_rate_history'].append(latest_data['event_count'])

                # 限制历史大小
                if len(self.monitor_data['timestamps']) > history_size:
                    self.monitor_data['timestamps'].pop(0)
                    self.monitor_data['angle_history'].pop(0)
                    self.monitor_data['control_history'].pop(0)
                    self.monitor_data['event_rate_history'].pop(0)

            # 生成监控图像
            monitor_image = self._create_monitor_image()

            # 显示图像
            cv2.imshow(self.window_name, monitor_image)
            cv2.waitKey(1)

            time.sleep(0.05)  # 20Hz更新率

    def _create_monitor_image(self):
        """创建监控图像"""
        width, height = 800, 600
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # 绘制标题
        cv2.putText(image, "倒立摆闭环系统实时监控", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # 绘制角度历史图
        if self.monitor_data['angle_history']:
            self._plot_history(
                image,
                self.monitor_data['angle_history'],
                "角度 (deg)",
                (50, 100),
                (350, 200),
                color=(0, 255, 0)
            )

        # 绘制控制力历史图
        if self.monitor_data['control_history']:
            self._plot_history(
                image,
                self.monitor_data['control_history'],
                "控制力 (N)",
                (450, 100),
                (350, 200),
                color=(255, 0, 0)
            )

        # 绘制事件率历史图
        if self.monitor_data['event_rate_history']:
            self._plot_history(
                image,
                self.monitor_data['event_rate_history'],
                "事件率",
                (50, 350),
                (350, 200),
                color=(255, 255, 0)
            )

        # 显示系统状态
        status_y = 320
        cv2.putText(image, "系统状态:", (450, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        status_color = (0, 255, 0) if self.system.running else (0, 0, 255)
        status_text = "运行中" if self.system.running else "已停止"
        cv2.putText(image, f"状态: {status_text}", (470, status_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)

        if self.monitor_data['timestamps']:
            current_time = self.monitor_data['timestamps'][-1]
            cv2.putText(image, f"时间: {current_time:.2f}s", (470, status_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if len(self.monitor_data['angle_history']) > 0:
                current_angle = np.degrees(self.monitor_data['angle_history'][-1])
                cv2.putText(image, f"当前角度: {current_angle:.1f}°", (470, status_y + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if len(self.monitor_data['control_history']) > 0:
                current_force = self.monitor_data['control_history'][-1]
                cv2.putText(image, f"当前控制力: {current_force:.2f}N", (470, status_y + 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 绘制控制示意图
        self._draw_control_diagram(image, (450, 380), (350, 200))

        return image

    def _plot_history(self, image, data, title, start_pos, size, color):
        """绘制历史数据图"""
        x_start, y_start = start_pos
        width, height = size

        if len(data) < 2:
            return

        # 绘制边框
        cv2.rectangle(image,
                      (x_start, y_start),
                      (x_start + width, y_start + height),
                      (100, 100, 100), 1)

        # 绘制标题
        cv2.putText(image, title, (x_start + 10, y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 计算数据范围
        data_min = min(data)
        data_max = max(data)
        data_range = data_max - data_min

        if data_range < 1e-6:
            data_range = 1.0

        # 绘制数据
        points = []
        for i, value in enumerate(data):
            x = x_start + int(i * width / (len(data) - 1))
            y = y_start + height - int((value - data_min) * height / data_range)
            points.append((x, y))

        # 绘制折线
        for i in range(1, len(points)):
            cv2.line(image, points[i - 1], points[i], color, 2)

        # 绘制零线
        if data_min <= 0 <= data_max:
            zero_y = y_start + height - int((0 - data_min) * height / data_range)
            cv2.line(image,
                     (x_start, zero_y),
                     (x_start + width, zero_y),
                     (150, 150, 150), 1)

    def _draw_control_diagram(self, image, start_pos, size):
        """绘制控制示意图"""
        x_start, y_start = start_pos
        width, height = size

        # 绘制标题
        cv2.putText(image, "控制系统示意图", (x_start + 10, y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 绘制框图
        box_width = 80
        box_height = 40
        padding = 20

        # 倒立摆
        cv2.rectangle(image,
                      (x_start + padding, y_start + padding),
                      (x_start + padding + box_width, y_start + padding + box_height),
                      (0, 200, 0), 2)
        cv2.putText(image, "倒立摆",
                    (x_start + padding + 10, y_start + padding + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 事件相机
        cv2.rectangle(image,
                      (x_start + padding, y_start + padding + box_height + padding),
                      (x_start + padding + box_width, y_start + padding + box_height * 2 + padding),
                      (200, 200, 0), 2)
        cv2.putText(image, "事件相机",
                    (x_start + padding + 10, y_start + padding + box_height + padding + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 状态估计器
        cv2.rectangle(image,
                      (x_start + padding + box_width + padding, y_start + padding + box_height + padding),
                      (x_start + padding + box_width * 2 + padding, y_start + padding + box_height * 2 + padding),
                      (200, 0, 200), 2)
        cv2.putText(image, "状态估计",
                    (x_start + padding + box_width + padding + 10, y_start + padding + box_height + padding + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 控制器
        cv2.rectangle(image,
                      (x_start + padding + box_width * 2 + padding * 2, y_start + padding + box_height + padding),
                      (x_start + padding + box_width * 3 + padding * 2, y_start + padding + box_height * 2 + padding),
                      (0, 0, 200), 2)
        cv2.putText(image, "控制器",
                    (x_start + padding + box_width * 2 + padding * 2 + 15,
                     y_start + padding + box_height + padding + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 绘制箭头
        arrow_color = (200, 200, 200)

        # 倒立摆 -> 事件相机
        self._draw_arrow(image,
                         (x_start + padding + box_width // 2, y_start + padding + box_height),
                         (x_start + padding + box_width // 2, y_start + padding + box_height + padding),
                         arrow_color)

        # 事件相机 -> 状态估计器
        self._draw_arrow(image,
                         (x_start + padding + box_width, y_start + padding + box_height + padding + box_height // 2),
                         (x_start + padding + box_width + padding,
                          y_start + padding + box_height + padding + box_height // 2),
                         arrow_color)

        # 状态估计器 -> 控制器
        self._draw_arrow(image,
                         (x_start + padding + box_width * 2 + padding,
                          y_start + padding + box_height + padding + box_height // 2),
                         (x_start + padding + box_width * 2 + padding * 2,
                          y_start + padding + box_height + padding + box_height // 2),
                         arrow_color)

        # 控制器 -> 倒立摆
        self._draw_arrow(image,
                         (x_start + padding + box_width * 3 + padding * 2 - box_width // 2,
                          y_start + padding + box_height + padding + box_height // 2),
                         (x_start + padding + box_width * 3 + padding * 2 - box_width // 2,
                          y_start + padding + box_height),
                         arrow_color)

        # 绘制反馈标签
        cv2.putText(image, "反馈",
                    (x_start + padding + box_width * 3 + padding * 2 - 25,
                     y_start + padding + box_height + padding // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def _draw_arrow(self, image, start, end, color, thickness=2):
        """绘制箭头"""
        cv2.arrowedLine(image, start, end, color, thickness, tipLength=0.1)