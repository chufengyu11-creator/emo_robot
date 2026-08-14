# 大模型链路与机器人技能测试

数据链路：

```text
机器人 10.0.1.41
  -> Windows 中继 10.0.1.99:18766
  -> 大模型服务器 111.56.189.29:8764
  -> OpenAI 兼容接口 127.0.0.1:8004/v1
```

按“服务器 → Windows 电脑 → 机器人”的顺序启动。

## 1. 启动服务器

在服务器执行：

```bash
cd /data

python3 tcp_json_prompt_server.py \
  --host 0.0.0.0 \
  --port 8764 \
  --openai-base-url http://127.0.0.1:8004/v1 \
  --model qwen \
  --max-tokens 256 \
  --temperature 0.0 \
  --timeout-s 120 \
  --disable-thinking
```

确保防火墙允许 Windows 电脑访问 TCP 端口 `8764`。

## 2. 启动 Windows 中继

在 Windows Anaconda Prompt 或 PowerShell 中执行：

```powershell
cd E:\personal\Emotion_ai 

python tcp_json_prompt_relay.py --listen-host 0.0.0.0 --listen-port 18766 --server-host 111.56.189.29 --server-port 8764 --timeout-s 130
```

Windows 防火墙需允许 Python 监听 TCP 端口 `18766`。可用以下命令确认监听状态：

```powershell
netstat -ano | findstr :18766
```

## 3. 在机器人上测试完整 TCP 链路

```bash
cd /home/agi/Documents/X2_Josh/emo_robot/scripts

python3 tcp_json_prompt_client.py 10.0.1.99 \
  --relay-port 18766 \
  --timeout-s 140 \
  --prompt "用户说：你好机器人。请用一句自然中文回应，不超过15个字。"
```

若出现 `OSError: [Errno 113] No route to host`，表示机器人尚未连到 Windows 电脑，错误发生在 TCP 请求和大模型推理之前。依次检查：

```bash
ip route get 10.0.1.99
ping -c 3 10.0.1.99
```

同时确认 Windows 当前地址确实为 `10.0.1.99`、两台设备处于同一局域网，并且 Windows 防火墙已放行 `18766`。

## 4. 编译并启动 ROS 2 Agent

每个机器人终端先加载相同的 ROS 2、AimDK 和 Conda 环境，然后进入实际工作空间。例如：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate emobot
source /opt/ros/humble/setup.bash
source ~/aimdk/install/local_setup.bash
cd /home/agi/Documents/emo_robot
```

编译并加载：

```bash
colcon build --symlink-install --packages-select emo_robot_agent
source install/local_setup.bash
```

启动 Agent，并覆盖当前配置文件中不同的默认中继地址：

```bash
ros2 launch emo_robot_agent language_agent.launch.py \
  llm_host:=10.0.1.99 \
  llm_port:=18766
```

默认 `execution_mode` 为 `simulation`，不会调用真实 AimDK 动作或 TTS。

## 5. 测试技能执行

另开一个已加载环境和工作空间的终端，发布测试计划：

```bash
ros2 topic pub --once \
  /emo_robot/agent/skill_plan \
  std_msgs/msg/String \
  "{data: '{\"tasks\":[\"wave\",\"bow\",\"clap\"]}'}"
```

查看技能执行状态：

```bash
ros2 topic echo /emo_robot/agent/status
```

查看 Planner 生成的计划：

```bash
ros2 topic echo /emo_robot/agent/skill_plan
```

查看最终识别文本

```bash
ros2 topic echo /emo_robot/agent/text_input
```

测试自然语言到技能计划的完整链路：

```bash
ros2 topic pub --once \
  /emo_robot/agent/text_input \
  std_msgs/msg/String \
  "{data: '挥手、鞠躬并鼓掌'}"
```

