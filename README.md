# Robot Task Playbook

移动机器人任务执行、视觉识别与故障处理的工程笔记，附两条实现路线。

这套逻辑整理自一次取得不错成绩的无人车任务挑战实战。这里更关心可复用的技术：怎样拆分任务、减少识别等待、组织参数，以及在设备异常时留下能用的诊断信息。

## 两种模式

| | ROS闭环模式 | 定时路线模式 |
|---|---|---|
| 参考思路 | 沿用资源包常见的导航、检测、服务分层方式 | 保留现场直接运行的顺序执行链 |
| 运动依据 | move_base、定位与传感器反馈 | 距离/速度、转角/角速度换算持续时间 |
| 视觉 | 本地几何匹配，或自行接入外部视觉服务 | 相机新帧 → 视觉API → 校验 → 本地WAV |
| 适用前提 | 定位、导航和标定已经建立 | 受控环境、固定起点、路线经过实测 |
| 主要代价 | 依赖多，联调时间长 | 没有避障与到位反馈，误差会累积 |

**ROS模式是对官方接口风格的参考实现，不是官方代码镜像，也不代表两种模式均已通过实车验收。** 定时模式中2/3号位置不再次识别或瞄准，执行的只是配置中的脉冲流程，不是自动命中系统。

## 先在电脑上看流程

Python 3.8及以上。建议使用虚拟环境；ROS部分还必须使用与目标ROS发行版匹配的Python绑定。

```bash
python -m pip install -r requirements.txt
python simulate_ros.py
python run_timed.py --plan config/timed_route.json
python run_timed.py --simulate --start-pose-test --plan config/timed_route.json --recognize-after-stop --camera 0 --shoot-and-retreat --continue-to-3
python tools/run_tests.py
```

`simulate_ros.py` 展示状态机，`run_timed.py` 不带模式参数时只预览计划。带 `--simulate` 的命令是离线模拟，不开相机、串口或API，依赖真实识别结果的后续动作会跳过；不能用它证明整场实车跑通。不要为运行示例添加 `--live`。

## 从哪里读

- [两条实现路线](docs/MODES.md)：状态机、现场顺序、入口与参数。
- [技术复盘](docs/ENGINEERING.md)：降低视觉等待、分离控制状态、排查串口与部署环境。
- [资源准备](docs/RESOURCES.md)：自备题卡和音频，填写自己的密钥，接入ROS服务。
- [验证范围](docs/VALIDATION.md)：只记录本发布副本实际跑过的检查。
- [来源与发布范围](PROVENANCE.md)：没有收录哪些材料，代码许可状态。

核心状态机在 `src/aic_competition/src/aic_competition/core.py`，定时执行在 `src/aic_competition/tools/fixed_route.py`，距离集中在 `config/timed_route.json`。模型配置在 `src/aic_competition/src/aic_competition/vision_config.py`；其中默认值记录的是当时的接口配置，不保证你的账号当前可用。

## 不包含什么

没有厂家驱动、官方资源包、比赛PDF及截图、题卡图集、场地地图、实拍数据、预生成语音、私人日志、历史凭据或构建缓存。缺少这些文件不是压缩包损坏。请按资源说明使用自己拥有权利的材料补齐，不要直接打包别人的资源。

请求接受、软件脉冲结束、机构实际动作和命中是不同层次。串口写入成功、播放器返回成功、计划时间走完，都不自动成为物理结果。任何硬件联调都需要独立的停车、断能和现场看护手段；不要同时运行两套控制入口。
