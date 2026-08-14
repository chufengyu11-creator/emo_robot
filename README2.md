# 智远 X2 机器人运动模式查询与切换

> **真机安全警告**
>
> 模式切换会直接改变机器人关节的受力和控制方式。`PASSIVE_DEFAULT`、
> `DAMPING_DEFAULT` 等模式可能使机器人失去主动站立能力并倒下。
> 本文中的切换命令仅供了解；在没有吊装、人员扶持、急停和官方状态转换流程保障时，
> 不要在真机执行。

## 1. 加载 ROS 2 与 AimDK 环境

每个新终端先执行：

```bash
source /opt/ros/humble/setup.bash
source ~/aimdk/install/local_setup.bash
```

确认官方示例存在：

```bash
ros2 pkg executables py_examples | grep mc_action
```

正常应包含：

```text
py_examples get_mc_action
py_examples set_mc_action
```

## 2. 查询当前机器人模式（只读）

确认查询服务存在：

```bash
ros2 service list | grep GetMcAction
```

查询当前模式：

```bash
ros2 run py_examples get_mc_action
```

该命令调用只读服务：

```text
/aimdk_5Fmsgs/srv/GetMcAction
```

预期输出类似：

```text
Robot mode get successfully.
Mode name: STAND_DEFAULT
Mode status: 100
```

模式状态值：

| 状态值 | 名称 | 说明 |
|---:|---|---|
| `0` | `IDLE` | 当前模式处于空闲状态 |
| `100` | `RUNNING` | 当前模式正在运行 |
| `200` | `TRANSITION` | 正在切换模式，禁止继续下发切换或运动命令 |

如果一直显示服务不可用，检查：

```bash
ros2 service list | grep /aimdk_5Fmsgs/srv/GetMcAction
```

没有输出时不要尝试调用切换服务，应先确认 AimDK 和机器人控制系统是否正常启动。

## 3. 支持的基础运动模式

| 模式 | 缩写 | 说明 | 主要风险/使用条件 |
|---|---|---|---|
| `PASSIVE_DEFAULT` | `PD` | 关节零力矩、自由状态，不含末端执行器 | 机器人可能立即失去支撑并倒下 |
| `DAMPING_DEFAULT` | `DD` | 关节有阻尼，但不主动保持站立 | 仍可能倒下，需要安全支撑 |
| `JOINT_DEFAULT` | `JD` | 位置控制站立、关节锁定 | 可能产生位置调整，要求姿态和环境安全 |
| `STAND_DEFAULT` | `SD` | 主动发力并动态平衡 | 必须已经直立且双脚落地 |
| `LOCOMOTION_DEFAULT` | `LD` | 正常走跑模式 | 只用于完成运动开发安全准备后 |

AimDK v0.8.0 及以后，`STAND_DEFAULT` 与 `LOCOMOTION_DEFAULT`
会根据运动命令在内部协调，通常不需要在两者之间反复手动切换。

## 4. 切换机器人模式（高风险，仅供参考）

确认切换服务：

```bash
ros2 service list | grep SetMcAction
```

官方命令格式：

```bash
ros2 run py_examples set_mc_action <模式缩写>
```

不带参数运行时，程序会显示菜单并等待输入：

```bash
ros2 run py_examples set_mc_action
```

以下命令都会立即请求切换真机状态，现在不要执行：

```bash
# 零力矩模式：机器人可能立即倒下
ros2 run py_examples set_mc_action PD

# 阻尼模式：不主动维持站立
ros2 run py_examples set_mc_action DD

# 位控站立
ros2 run py_examples set_mc_action JD

# 稳定站立：执行前必须确保机器人直立且双脚落地
ros2 run py_examples set_mc_action SD

# 走跑模式
ros2 run py_examples set_mc_action LD
```

该示例调用：

```text
/aimdk_5Fmsgs/srv/SetMcAction
```

成功时输出：

```text
Robot mode set successfully.
```

切换请求返回成功后，仍应重新执行：

```bash
ros2 run py_examples get_mc_action
```

只有目标模式的 `Mode status` 变为 `100`（`RUNNING`），才能认为切换完成。
状态为 `200`（`TRANSITION`）时，不要继续下发模式或运动命令。

## 5. 真机切换前检查

执行任何 `set_mc_action` 命令前，必须同时满足：

1. 已查询并记录当前模式，确认目标转换符合官方状态转换图。
2. 机器人周围无人、无障碍物，地面平整。
3. 急停可立即触达，并有熟悉 X2 的人员在机器人旁监护。
4. 进入零力矩或阻尼模式时，机器人已有吊装或可靠的机械/人员支撑。
5. 进入稳定站立模式前，机器人已经直立，双脚完整接触地面。
6. 没有其他节点、遥控器或程序同时发送运动控制命令。
7. 模式处于 `TRANSITION` 时等待完成，不重复发送切换请求。

当前 `emo_robot` 的摄像头、WAVE/COME 检测和 TTS 测试不需要切换运动模式。
保持配置中的 `motion_enabled: false` 即可。

## 6. 站立后的原地转向与前进（高风险，仅供以后测试）

> **现在不要执行本节命令。**
>
> 下述官方示例会注册运动控制输入源，并向真机持续发布速度命令。
> 它不是只发送一次的“点动”测试：确认三项输入后，机器人会按设定速度运动约
> **5 秒**。终端、SSH 或网络异常都不能替代实体急停。

### 6.1 运动前的只读检查

未来准备进行运动测试时，先查询模式：

```bash
ros2 run py_examples get_mc_action
```

AimDK v0.8.0+ 将稳定站立与走跑模式设计为一体化状态机。只要查询结果满足
以下任意一种状态，就可以由运动控制器根据速度命令在内部自动切换：

```text
Mode name: STAND_DEFAULT
Mode status: 100
```

或者：

```text
Mode name: LOCOMOTION_DEFAULT
Mode status: 100
```

`emo_robot` 的安全测试节点接受上述两个 `RUNNING` 状态，但不会主动调用模式切换服务。
状态为 `TRANSITION(200)` 或处于其他模式时仍会拒绝运动。

再检查速度话题上是否已有其他发布者：

```bash
ros2 topic info /aima/mc/locomotion/velocity --verbose
```

还可以只读查询当前运动控制输入源：

```bash
ros2 run py_examples get_current_input_source
```

如果当前不是上述任一 `RUNNING(100)` 状态，或者存在未确认的控制输入源，
不要继续。
真机周围必须预留足够空间，并由一人在机器人旁监护、随时操作实体急停。

### 6.2 官方定时速度示例

启动官方交互示例的命令为：

```bash
ros2 run py_examples mc_locomotion_velocity
```

程序依次要求输入：

```text
前进速度：0 或 ±(0.2～1.0) m/s
侧移速度：0 或 ±(0.2～1.0) m/s
旋转速度：0 或 ±(0.1～1.0) rad/s
```

速度方向定义：

| 字段 | 正值 | 负值 | 单位 |
|---|---|---|---|
| 前进速度 | 前进 | 后退 | `m/s` |
| 侧移速度 | 左移 | 右移 | `m/s` |
| 旋转速度 | 原地左转 | 原地右转 | `rad/s` |

程序会以 50 Hz 发布速度，持续约 5 秒，然后将速度清零并继续运行。
测试结束后才使用 `Ctrl+C` 退出。`Ctrl+C` 和软件清零都不是安全急停，
发生失控或异常运动时应使用实体急停。

### 6.3 前进命令示例（不要现在执行）

运行交互示例后，三个提示依次输入：

```text
0.2
0
0
```

含义是以 `0.2 m/s` 向前运动约 5 秒。理论位移约为 **1 米**，
实际位移会受机器人控制、地面和加减速过程影响。因此，这不是小距离试走，
狭小空间内禁止使用。

### 6.4 原地转向命令示例（不要现在执行）

原地左转时，三个提示依次输入：

```text
0
0
0.1
```

原地右转时，三个提示依次输入：

```text
0
0
-0.1
```

`0.1 rad/s` 持续 5 秒的理论转角约为 `0.5 rad`，即约 **28.6°**；
真机实际转角可能不同。转向前也必须清空机器人四周区域。

### 6.5 键盘控制示例（风险更高，仅记录命令）

官方 SDK 还提供连续键盘控制：

```bash
ros2 run py_examples keyboard
```

按键说明：

| 按键 | 功能 |
|---|---|
| `W` / `S` | 增加前进速度 / 增加后退速度 |
| `A` / `D` | 增加左移速度 / 增加右移速度 |
| `Q` / `E` | 增加左转角速度 / 增加右转角速度 |
| `Space` | 将三项速度清零 |
| `Esc` | 退出程序 |

每按一次，线速度变化 `0.2 m/s`，角速度变化 `0.1 rad/s`。
速度会保持并持续发布，直到再次按键改变或按 `Space` 清零，所以它比定时示例
更容易造成持续运动。`Space` 只是软件零速命令，不是实体急停。

官方键盘示例可能还要求处理遥控器输入源占用。不要为了运行示例而贸然关闭遥控器，
否则可能失去重要的人工控制手段；必须先按 X2 官方真机测试流程确认输入源优先级、
遥控器和实体急停方案。

### 6.6 `emo_robot` 独立安全测试节点

项目已将官方速度控制封装为独立节点，但它不属于 `gesture_detector`、TTS 或总
launch，也不会被手势自动触发：

```text
ros2 run emo_robot_motion locomotion_test
```

重新构建并加载运动包：

```bash
cd ~/Documents/emo_robot

"$CONDA_PREFIX/bin/colcon" build \
  --symlink-install \
  --packages-select emo_robot_motion

source install/local_setup.bash
```

安全默认配置位于：

```text
src/emo_robot_motion/config/locomotion_test.yaml
```

只验证节点安装和安全锁，不注册输入源、不发布任何速度：

```bash
ros2 run emo_robot_motion locomotion_test \
  --ros-args \
  --params-file src/emo_robot_motion/config/locomotion_test.yaml
```

预期日志包含：

```text
Locomotion test is locked: motion_enabled=false. No input source was registered and no velocity was published.
```

公共参数：

| 参数 | 安全默认值 | 约束 |
|---|---:|---|
| `motion_enabled` | `false` | 必须显式设为 `true` 才可能运动 |
| `confirmation` | `""` | 运动时必须严格等于 `I_UNDERSTAND` |
| `forward_velocity` | `0.0` | `0` 或 `±0.2～1.0 m/s` |
| `lateral_velocity` | `0.0` | `0` 或 `±0.2～1.0 m/s` |
| `angular_velocity` | `0.0` | `0` 或 `±0.1～1.0 rad/s` |
| `duration_s` | `2.0` | 必须大于 0，最大 5 秒 |

初版每次最多允许一个速度轴非零。下面只是命令格式示例，**现在不要在真机执行**：

```bash
ros2 run emo_robot_motion locomotion_test \
  --ros-args \
  -p motion_enabled:=true \
  -p confirmation:=I_UNDERSTAND \
  -p forward_velocity:=0.0 \
  -p lateral_velocity:=0.0 \
  -p angular_velocity:=0.0 \
  -p duration_s:=1.0
```

```bash
# 高风险：满足全部安全准备后才可用于 0.2 m/s、2 秒的单轴前进测试
ros2 run emo_robot_motion locomotion_test \
  --ros-args \
  --params-file src/emo_robot_motion/config/locomotion_test.yaml \
  -p motion_enabled:=true \
  -p confirmation:=I_UNDERSTAND \
  -p forward_velocity:=0.2 \
  -p duration_s:=2.0
```

即使提供了解锁参数，节点仍会先只读调用 `GetMcAction`，仅当模式为
`STAND_DEFAULT(200)` 或 `LOCOMOTION_DEFAULT(300)`，并且状态为 `RUNNING(100)` 时
才注册输入源。随后以 50 Hz 发布单轴速度，到期或收到 Ctrl+C 后连续发布 3 帧
零速度、停止定时器、注销
`emo_robot_locomotion_test` 输入源并退出。查询失败、模式不符、注册失败和参数非法
都会拒绝运动；节点不会自动切换机器人模式。

不要把该节点加入 `gesture_approach.launch.py`，不要连接手势事件，也不要使用
`ros2 topic pub --once /aima/mc/locomotion/velocity ...` 绕过输入源注册、模式检查和
超时停车机制。软件停车不是实体急停，发生异常运动时必须使用实体急停。
