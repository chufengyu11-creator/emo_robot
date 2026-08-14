# emo_robot_asr

订阅AimDK降噪VAD音频，将完整语句上传到远程SenseVoice服务，并把最终文本发布给Agent。第一版需要先说X2唤醒词。

## 启动

```bash
source /opt/ros/humble/setup.bash
source ~/aimdk/install/local_setup.bash
source ~/Documents/emo_robot/install/local_setup.bash
ros2 run emo_robot_asr vad_asr --ros-args \
  --params-file ~/Documents/emo_robot/src/emo_robot_asr/config/asr.yaml
```

通常直接启动Agent即可，ASR默认随其启动：

```bash
ros2 launch emo_robot_agent language_agent.launch.py
```

确认麦克风和话题：
```bash
ros2 run py_examples get_mic_source
ros2 topic info /agent/process_audio_output --verbose
ros2 topic echo /emo_robot/asr/status
ros2 topic echo /emo_robot/agent/text_input
```

关闭Agent自动启动的ASR：
```bash
ros2 launch emo_robot_agent language_agent.launch.py enable_asr:=false
```

ASR输入为AimDK的16kHz、16-bit、单声道PCM；节点仅在VAD结束后上传内存WAV，不保存录音。
