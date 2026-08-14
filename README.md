# emo_robot

智远 X2 旗舰版的 ROS 2 Humble 手势召唤项目。系统通过外接 USB 摄像头识别
WAVE/COME，使用胸前 LiDAR 获取目标距离，并执行转向、对准和接近。

> **真机安全：** 当前配置可能控制机器人运动。首次运行应在空旷、平整、无障碍物的
> 场地进行，并确保实体急停可触达。只调试画面时请先关闭 `motion_enabled`。

## 项目目录

```text
emo_robot/
├── src/
│   ├── emo_robot_interfaces/   # 自定义 ROS 消息
│   ├── emo_robot_perception/   # 摄像头、手势和 LiDAR 感知
│   ├── emo_robot_control/      # 语音响应和接近控制
│   ├── emo_robot_motion/       # AimDK 运动客户端
│   ├── emo_robot_speech/       # AimDK TTS 客户端
│   ├── emo_robot_asr/          # AimDK降噪麦克风与远程ASR
│   ├── emo_robot_agent/        # LLM规划和预设动作执行
│   └── emo_robot_bringup/      # launch 和 YAML 参数
├── docs/                       # 设计文档
├── scripts/                    # 数据录制和辅助脚本
├── models/                     # 工作空间级实验模型
├── build/                      # colcon 生成
├── install/                    # colcon 生成
└── log/                        # colcon 生成
```

## 从 WSL 复制到 Jetson NX

项目应运行在 X2 的 Jetson NX 开发计算单元。下面示例中的 WSL 用户名和 Jetson IP
需要按实际环境替换：

```bash
rsync -av \
  --exclude .git \
  --exclude build \
  --exclude install \
  --exclude log \
  /home/ming/Documents/emo_robot/ \
  agi@10.0.1.41:/home/agi/Documents/emo_robot/
```

复制后检查模型：

```bash
ls -lh \
  ~/Documents/emo_robot/src/emo_robot_perception/models/yolo26n-pose.pt \
  ~/Documents/emo_robot/src/emo_robot_perception/models/yolo26n-pose-fp16.onnx
```

## 安装环境和依赖

ROS Humble 和 AimDK 需要预先安装。进入 Miniforge 的 `emobot` 环境：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate emobot
source /opt/ros/humble/setup.bash
source ~/aimdk/install/local_setup.bash

which python
which colcon
```

预期 Python 路径：

```text
/home/agi/miniforge3/envs/emobot/bin/python
```

安装系统工具：

```bash
sudo apt update
sudo apt install -y ros-humble-web-video-server v4l-utils
```

安装构建和通用视觉依赖：

```bash
python -m pip install \
  colcon-common-extensions==0.3.0 \
  setuptools==79.0.1 \
  empy==3.3.4 \
  lark==1.1.9 \
  numpy==1.26.4 \
  opencv-python==4.10.0.84 \
  Pillow==12.3.0 \
  mediapipe==0.10.18
```

Torch、TorchVision 和 ONNX Runtime GPU 必须安装与 JetPack 6、CUDA 和 aarch64
匹配的 NVIDIA/Jetson wheel。不要使用普通 `pip install torch` 覆盖 Jetson 版本。
确认这些 GPU 包可用后，再安装 Ultralytics：

```bash
python -c "import torch, torchvision, onnxruntime; print(torch.__version__, torchvision.__version__, onnxruntime.__version__)"
python -m pip install ultralytics==8.4.107
```

快速检查运行环境：

```bash
python - <<'PY'
import cv2
import mediapipe
import onnxruntime as ort
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mediapipe.__version__)
print("ONNX providers:", ort.get_available_providers())
PY
```

ONNX GPU 推理需要看到 `CUDAExecutionProvider`。

## 编译构建

```bash
cd ~/Documents/emo_robot
source ~/miniforge3/etc/profile.d/conda.sh
conda activate emobot
source /opt/ros/humble/setup.bash
source ~/aimdk/install/local_setup.bash

"$CONDA_PREFIX/bin/colcon" build \
  --symlink-install \
  --cmake-args \
  -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"

source install/local_setup.bash
```

修改代码后的增量构建：

```bash
"$CONDA_PREFIX/bin/colcon" build \
  --symlink-install \
  --packages-up-to emo_robot_bringup \
  --cmake-args \
  -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"

source install/local_setup.bash
```

## 启动

每个新终端先加载环境：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate emobot
source /opt/ros/humble/setup.bash
source ~/aimdk/install/local_setup.bash
source ~/Documents/emo_robot/install/local_setup.bash
```

当前 YAML 中 `motion_enabled: true` 时，总 launch 可能控制真机。只调试感知画面时，
先在 `src/emo_robot_bringup/config/gesture_approach.yaml` 中设置：

```yaml
approach_controller:
  ros__parameters:
    motion_enabled: false
    confirmation: ""
```

启动完整流程：

```bash
ros2 launch emo_robot_bringup gesture_approach.launch.py
```

### 单独启动节点

格式：

```bash
ros2 run <功能包> <可执行名> --ros-args --params-file <YAML路径>
```

例如只启动手势识别节点：

```bash
ros2 run emo_robot_perception gesture_detector \
  --ros-args \
  --params-file ~/Documents/emo_robot/src/emo_robot_bringup/config/gesture_approach.yaml
```

例如只启动 `person_tracker`：

```bash
ros2 run emo_robot_perception person_tracker \
  --ros-args \
  --params-file ~/Documents/emo_robot/src/emo_robot_bringup/config/gesture_approach.yaml
```

`ros2 run` 只启动指定节点，不会自动启动摄像头、LiDAR 或其他上游节点。

## 查看调试画面

另开一个 X2 终端，加载 ROS 和项目环境后启动视频服务：

```bash
source /opt/ros/humble/setup.bash
source ~/Documents/emo_robot/install/local_setup.bash
ros2 run web_video_server web_video_server
```

查询 X2 的局域网 IP：

```bash
hostname -I
```

电脑与 X2 在同一局域网时，直接复制以下地址到浏览器：

```text
http://10.0.1.41:8080/
http://10.0.1.41:8080/stream?topic=/emo_robot/perception/debug_image
http://10.0.1.41:8080/stream?topic=/emo_robot/camera/image_raw
http://10.0.1.41:8080/stream?topic=/emo_robot/lidar/target_debug_image
```

终端检查：

```bash
# 查看全部节点和话题
ros2 node list
ros2 topic list

# 摄像头和手势调试图频率
ros2 topic hz /emo_robot/camera/image_raw
ros2 topic hz /emo_robot/perception/debug_image

# WAVE / COME 手势事件
ros2 topic echo /emo_robot/perception/gesture_detection

# LiDAR 检测到的目标簇
ros2 topic echo /emo_robot/perception/lidar_target

# person_tracker 输出的融合人物目标
ros2 topic echo /emo_robot/perception/person_target

# approach_controller 计算的前进、横移和转向速度
ros2 topic echo /emo_robot/control/approach_cmd_debug
```
