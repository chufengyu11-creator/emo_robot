# emo_robot_agent

默认launch自动加载 `config/agent.yaml`：planner使用 `tcp_llm`，预设动作使用 `motion_mode=simulation`，语音使用 `tts_mode=real`。因此默认不会控制机器人动作，但会调用X2真实TTS。TCP服务不可用时不会回退到demo。

数据链路：AimDK降噪麦克风 → 远程ASR → 语言规划 → 顺序技能 → AimDK预设动作 → TTS。默认launch会启动 `emo_robot_asr`；需要先说X2唤醒词。

## 1. 编译

```bash
cd ~/Documents/emo_robot
colcon build --symlink-install --packages-select emo_robot_agent
```

## 2. 加载

```bash
source install/local_setup.bash
```

## 3. 启动

```bash
ros2 launch emo_robot_agent language_agent.launch.py
```

只测试手动文本输入、不启动ASR：

```bash
ros2 launch emo_robot_agent language_agent.launch.py enable_asr:=false
```

默认的“模拟动作＋真实TTS”也可以显式写成：

```bash
ros2 launch emo_robot_agent language_agent.launch.py \
  planner_backend:=demo \
  motion_mode:=simulation \
  tts_mode:=real \
  motion_enabled:=false
```

动作和TTS全部模拟：

```bash
ros2 launch emo_robot_agent language_agent.launch.py \
  planner_backend:=demo \
  motion_mode:=simulation \
  tts_mode:=simulation \
  motion_enabled:=false
```

真机动作必须在空旷场地、机器人稳定站立且急停可触达时显式解锁：

```bash
ros2 launch emo_robot_agent language_agent.launch.py \
  planner_backend:=demo \
  motion_mode:=real \
  tts_mode:=real \
  motion_enabled:=true \
  confirmation:=I_UNDERSTAND
```

每个动作下发前都会确认机器人处于 `STAND_DEFAULT/RUNNING`。默认仍是仿真模式。

`motion_mode`和`tts_mode`只接受 `simulation` 或 `real`。旧参数 `execution_mode` 已删除。`motion_mode=simulation` 时禁止设置 `motion_enabled=true`。

参数优先级：显式 launch 参数 > `config/agent.yaml` > 节点代码默认值。

## 4. 测试技能执行

另开终端并加载工作空间后执行：

```bash
ros2 topic pub --once \
  /emo_robot/agent/skill_plan \
  std_msgs/msg/String \
  "{data: '{\"tasks\":[\"wave_right\",\"bow\",\"clap\"]}'}"
```

## 5. 查看执行状态

```bash
ros2 topic echo /emo_robot/agent/status
```

也可以查看 planner 生成的计划：

```bash
ros2 topic echo /emo_robot/agent/skill_plan
```

发送自然语言给离线 planner：

```bash
ros2 topic pub --once \
  /emo_robot/agent/text_input \
  std_msgs/msg/String \
  "{data: '挥手、鞠躬并鼓掌'}"
```

## 6. ASR接口

默认ASR服务为 `http://111.56.189.29:8765/v1/audio/transcriptions`。检查识别状态：

```bash
ros2 run py_examples get_mic_source
ros2 topic echo /emo_robot/asr/status
ros2 topic echo /emo_robot/agent/text_input
```

ASR只发布一句完整的最终识别文本，不发布逐字变化的中间结果：

```text
topic: /emo_robot/agent/text_input
type:  std_msgs/msg/String
data:  一次完整的最终识别文本
```

输入话题可通过 `language_planner.input_topic` 参数修改。空文本会被拒绝；执行器忙碌时，新计划会在 `/emo_robot/agent/status` 中返回 `plan_failed`。

计划允许 `tasks: []` 的纯语音回复；只要 `speech_text` 非空，就会按 `tts_mode` 播放。`introduce` 本身不再单独播报，避免与计划末尾的 `speech_text` 重复。

## 7. 真机动作映射

| 技能 | area/motion | 技能 | area/motion |
|---|---|---|---|
| wave_right | 2/1002 | wave_left | 1/1002 |
| handshake_right | 2/1003 | handshake_left | 1/1003 |
| raise_hand_right | 2/1001 | raise_hand_left | 1/1001 |
| flying_kiss_right | 2/1004 | flying_kiss_left | 1/1004 |
| salute_right | 2/1013 | salute_left | 1/1013 |
| heart_both | 3/1007 | heart_right / heart_left | 2/1007、1/1007 |
| flat_raise_both | 3/1010 | flat_raise_right / flat_raise_left | 2/1010、1/1010 |
| high_five_right | 2/1008 | high_five_left | 1/1008 |
| chest_wave_right | 2/1011 | chest_wave_left | 1/1011 |
| clap | 11/3017 | hug | 11/3008 |
| cheer | 11/3011 | bye | 11/3031 |
| dynamic_wave | 11/3007 | cross_arms | 11/3009 |
| bow | 11/3001 | scratch_head | 11/3024 |
| scratch_butt | 11/3025 |  |  |

未指定左右手时，挥手、握手、举手、飞吻、敬礼和击掌默认右手；比心和平举默认双手。旧技能名 `wave`、`handshake`、`salute`、`heart`、`raise_hand` 已移除。

AimDK只确认预设动作请求被接受，没有按 `task_id` 查询动作结束的接口，因此本包按YAML中的固定时间等待后再执行下一技能。所有动作完成后才播放计划中的 `speech_text`。
