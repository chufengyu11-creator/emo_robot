# 手势召唤与自主接近：初始设计

## 数据流

```text
RGB 图像
  -> gesture_detector
  -> /emo_robot/perception/gesture_detection
  -> person_tracker + 深度图
  -> /emo_robot/perception/person_target
  -> approach_controller
  -> AimDK 走跑速度接口（后续接入）
```

## 状态机

- `IDLE`：等待有效招手。
- `CONFIRMING`：跨多帧确认招手，降低误触发。
- `ALIGNING`：原地转向，让目标处于画面中央。
- `APPROACHING`：持续锁定同一人物并接近。
- `STOPPED`：达到停车距离。
- `TARGET_LOST`：目标丢失，立即清零速度。
- `EMERGENCY_STOP`：障碍、传感器或控制异常，持续清零速度。

## 安全约束

- 真实运动默认禁用。
- 控制节点退出、目标超时或深度无效时必须下发零速度。
- 走跑前必须注册唯一的 AimDK 输入源，例如 `emo_gesture_approach`。
- 必须限制线速度、角速度及加速度，避免速度突变。
- 停车距离需留出人体晃动和深度噪声余量，初始建议 1.2～1.5 米。
- USB 图像应尽量在摄像头所在计算单元处理；普通 USB RGB 相机不提供深度。
- 真实机器人测试前先完成仿真和架空/保护测试。

## 激光雷达输入

```text
胸前 LiDAR PointCloud2 ─┐
                        ├─> lidar_subscriber ─> 后续障碍检测/安全停车
胸前 LiDAR IMU ─────────┘
```

当前 `lidar_subscriber` 只验证和汇报数据，没有向运动控制节点发布障碍结果。接入自主接近前，应定义独立的障碍状态消息，并让控制节点在目标丢失或障碍过近时优先停车。
