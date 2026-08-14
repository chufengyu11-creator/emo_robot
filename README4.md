# X2 自带语音交互的关闭、重启与恢复

本文记录在 X2 Ultra 上临时关闭自带大模型交互、保留系统降噪音频流，以及重启或恢复自带语音交互的方法。

## 1. 计算单元说明

X2 Ultra 的相关计算单元：

| 计算单元 | IP | 用途 |
|---|---|---|
| PC2 开发计算单元 | `10.0.1.41` | 运行 AimDK、ROS 2 和二次开发程序 |
| PC3 交互计算单元 | `10.0.1.42` | 运行机器人自带的 `agent` 交互服务 |

`SetAgentPropertiesRequest` 可以在 PC2 上通过 ROS 2 调用，但 `aima em stop-app agent` 和 `aima em start-app agent` 必须登录 PC3 后执行。在 PC2 上找不到 `agent` 是正常现象。

## 2. 切换为 only_voice 模式

`only_voice` 会关闭自带大模型/自然语言对话，但保留麦克风采集、系统降噪和 VAD 音频输出，适合接入自研 ASR。

在 PC2（`10.0.1.41`）加载环境：

```bash
source /opt/ros/humble/setup.bash
source ~/aimdk/install/local_setup.bash
```

设置运行模式：

```bash
ros2 service call \
  /aimdk_5Fmsgs/srv/SetAgentPropertiesRequest \
  aimdk_msgs/srv/SetAgentPropertiesRequest "
contents:
  properties:
    - key:
        value: 2  # AGENT_PROPERTY_RUN_MODE
      value: 'only_voice'
"
```

成功响应应包含：

```text
code=0
status.value=0
```

该调用只写入运行模式配置，必须重启 PC3 上的 `agent` 才会生效。

> 不要设置为 `no_voice`。`no_voice` 会完全禁用 agent，系统也不会提供降噪后的音频流。

## 3. 在 PC3 重启 agent

从 PC2 或同一机器人网络中的电脑登录 PC3：

```bash
ssh <PC3用户名>@10.0.1.42
```

确认当前确实位于 PC3，并检查应用：

```bash
hostname
aima em list-apps | grep agent
```

停止自带交互服务：

```bash
aima em stop-app agent
```

以刚设置的 `only_voice` 模式重新启动：

```bash
aima em start-app agent
```

如果只是需要让已保存的配置重新生效，也可以使用：

```bash
aima em restart-app agent
```

优先采用官方文档给出的 `stop-app`、`start-app` 两步操作。

## 4. 验证降噪音频流

回到 PC2（`10.0.1.41`），重新加载 ROS 2 与 AimDK 环境：

```bash
source /opt/ros/humble/setup.bash
source ~/aimdk/install/local_setup.bash
```

检查降噪音频话题：

```bash
ros2 topic info /agent/process_audio_output --verbose
```

运行官方接收示例：

```bash
cd ~/Documents
ros2 run py_examples mic_receiver
```

AimDK v0.9+ 的 `only_voice` 模式通常需要第一次说唤醒词来长期激活 VAD。之后说话时应看到：

```text
VAD state: Speech start
VAD state: Speech in progress
VAD state: Speech end
Audio segment saved: ...
```

降噪音频格式为 PCM、16 kHz、16 bit、单声道。

## 5. 仅停止和重新启动 agent

以下命令必须在 PC3（`10.0.1.42`）执行。

停止：

```bash
aima em stop-app agent
```

重新启动：

```bash
aima em start-app agent
```

停止期间系统降噪、VAD、自带语音交互以及相关 agent 音频话题可能都不可用。需要自研 ASR 使用系统降噪音频时，不应让 agent 长期保持停止状态，而应将其设置为 `only_voice` 后重新启动。

## 6. 恢复自带语音交互

在 PC2 将运行模式恢复为 `normal`：

```bash
source /opt/ros/humble/setup.bash
source ~/aimdk/install/local_setup.bash

ros2 service call \
  /aimdk_5Fmsgs/srv/SetAgentPropertiesRequest \
  aimdk_msgs/srv/SetAgentPropertiesRequest "
contents:
  properties:
    - key:
        value: 2  # AGENT_PROPERTY_RUN_MODE
      value: 'normal'
"
```

确认响应包含 `code=0` 和 `status.value=0` 后，登录 PC3 重启 agent：

```bash
ssh <PC3用户名>@10.0.1.42
aima em stop-app agent
aima em start-app agent
```

完成后，机器人恢复自带大模型、语音识别和自然语言交互。

## 7. 常见问题

### PC2 执行 stop-app 时提示 agent 不存在

这是因为 `agent` 位于 PC3。不要在 PC2 提供的应用列表中随意选择 `auto_integrated`、`perception` 或其他应用代替，应该登录 `10.0.1.42` 后操作。

### 设置 only_voice 后仍有原来的交互行为

属性服务只保存配置。确认调用返回成功后，还需要在 PC3 重启 `agent`。

### process_audio_output 没有音频

依次检查：

```bash
# PC3：agent 是否存在并已经启动
aima em list-apps | grep agent

# PC2：话题是否存在发布者
ros2 topic info /agent/process_audio_output --verbose

# PC2：运行官方接收程序
ros2 run py_examples mic_receiver
```

在 AimDK v0.9+ 中，首次使用 `only_voice` 时还需要说一次唤醒词来激活 VAD。
