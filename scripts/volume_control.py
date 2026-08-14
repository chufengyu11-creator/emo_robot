#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 查询音量
# ros2 service call /aimdk_5Fmsgs/srv/GetVolume aimdk_msgs/srv/GetVolume "{request: {}}"
# python volume_control.py get
# 设置音量
# ros2 service call /aimdk_5Fmsgs/srv/SetVolume aimdk_msgs/srv/SetVolume "{request: {}, audio_volume: 50}"
# python volume_control.py set 50

import rclpy
from rclpy.node import Node
from aimdk_msgs.srv import SetVolume, GetVolume
import sys

class VolumeController(Node):
    def __init__(self):
        super().__init__('volume_controller')
        self.set_client = self.create_client(SetVolume, '/aimdk_5Fmsgs/srv/SetVolume')
        self.get_client = self.create_client(GetVolume, '/aimdk_5Fmsgs/srv/GetVolume')
        
    def set_volume(self, volume):
        """设置音量"""
        if not self.set_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('SetVolume 服务不可用')
            return False
            
        request = SetVolume.Request()
        request.audio_volume = volume
        
        future = self.set_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info(f'✓ 音量已设置为: {volume}%')
            return True
        else:
            self.get_logger().error('✗ 设置音量失败')
            return False
    
    def get_volume(self):
        """获取当前音量"""
        if not self.get_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('GetVolume 服务不可用')
            return None
            
        request = GetVolume.Request()
        future = self.get_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            return future.result().audio_volume
        return None

def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python3 volume_control.py set <音量>  # 设置音量 (0-100)")
        print("  python3 volume_control.py get         # 获取当前音量")
        print("  python3 volume_control.py up          # 音量+10")
        print("  python3 volume_control.py down        # 音量-10")
        print("  python3 volume_control.py mute        # 静音")
        print("\n示例:")
        print("  python3 volume_control.py set 50")
        print("  python3 volume_control.py get")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    rclpy.init()
    controller = VolumeController()
    
    try:
        if command == 'set':
            if len(sys.argv) < 3:
                print("错误: 请指定音量值 (0-100)")
                sys.exit(1)
            volume = int(sys.argv[2])
            if volume < 0 or volume > 100:
                print("错误: 音量必须在 0-100 之间")
                sys.exit(1)
            controller.set_volume(volume)
            
        elif command == 'get':
            volume = controller.get_volume()
            if volume is not None:
                print(f"当前音量: {volume}%")
                
        elif command == 'up':
            current = controller.get_volume()
            if current is not None:
                new_volume = min(current + 10, 100)
                controller.set_volume(new_volume)
                
        elif command == 'down':
            current = controller.get_volume()
            if current is not None:
                new_volume = max(current - 10, 0)
                controller.set_volume(new_volume)
                
        elif command == 'mute':
            controller.set_volume(0)
            print("已静音")
            
        else:
            print(f"未知命令: {command}")
            sys.exit(1)
            
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()