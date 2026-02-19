# integrated_event_camera.py
"""
集成的事件相机模拟器，与倒立摆模拟器协同工作
"""
import numpy as np
import cv2
import sys

sys.path.append("../src")

from dvs_sensor import DvsSensor
from event_buffer import EventBuffer
from event_display import EventDisplay


class IntegratedEventCamera:
    """集成的事件相机模拟器"""

    def __init__(self, width, height, config=None):
        """
        初始化事件相机模拟器

        Args:
            width: 图像宽度
            height: 图像高度
            config: DVS配置参数
        """
        self.width = width
        self.height = height

        # DVS默认配置
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
            'dt': 1000,  # 微秒
        }

        if config:
            self.config.update(config)

        # 初始化DVS传感器
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

        # 事件缓冲区
        self.event_buffer = EventBuffer(1000)

        # 事件显示器
        render_timesurface = 1
        self.event_display = EventDisplay(
            "Event Camera Output",
            width, height,
            self.config['dt'],
            render_timesurface
        )

        # 时间管理
        self.current_time_us = 0
        self.frame_count = 0

        # 事件统计
        self.event_count = 0
        self.event_rate_history = []

        print(f"集成事件相机初始化完成: {width}x{height}")

    def init_with_frame(self, frame):
        """
        使用第一帧初始化DVS

        Args:
            frame: 第一帧图像 (BGR格式)
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 转换为DVS输入格式
        dvs_input = gray / 255.0 * 1e4

        # 初始化DVS
        self.dvs.init_image(dvs_input)

        print("DVS使用第一帧图像初始化完成")

    def process_frame(self, frame, dt_us=None):
        """
        处理一帧图像，生成事件

        Args:
            frame: 当前帧图像 (BGR格式)
            dt_us: 时间间隔 (微秒)，如果为None则使用配置中的dt

        Returns:
            事件缓冲区
        """
        if dt_us is None:
            dt_us = self.config['dt']

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 转换为DVS输入格式
        dvs_input = gray / 255.0 * 1e4

        # 生成事件
        events = self.dvs.update(dvs_input, dt_us)

        # 更新时间
        self.current_time_us += dt_us
        self.frame_count += 1

        # 更新事件统计
        self.event_count += events.i
        self.event_rate_history.append(events.i / (dt_us * 1e-6))  # 事件/秒

        # 显示事件
        self.event_display.update(events, dt_us)

        return events

    def get_event_statistics(self):
        """获取事件统计信息"""
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
        """重置事件相机"""
        # 重新初始化DVS
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

        # 重置状态
        self.current_time_us = 0
        self.frame_count = 0
        self.event_count = 0
        self.event_rate_history = []

        print("事件相机已重置")