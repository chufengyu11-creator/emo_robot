# Models

此目录用于保存工作空间级实验模型。

真机使用的 `yolo26n-pose.pt` 已放入
`src/emo_robot_perception/models/`，构建时会安装到
`share/emo_robot_perception/models/`。节点默认从 ROS package share
读取该模型，不依赖固定的 `/home/agi` 路径，也不会在运行时联网下载。

根目录下的其他大模型仍由 `.gitignore` 排除。建议记录模型来源、版本、
输入尺寸、归一化方式、许可证和导出命令，避免以后无法复现实验。
