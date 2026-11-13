# 两条实现路线

## 1. ROS闭环模式

这条路线保留导航与任务控制的分层结构：

```text
定位 / move_base / 目标检测
            ↓
RosIO → CompetitionMission → 任务状态
            ↓
安全桥 → 速度仲裁 / 发射执行状态
```

`core.py` 不依赖ROS。`competition_node.py` 将导航、检测、视觉服务、播音和任务状态接起来；`safety_bridge.py` 管理速度租约、急停状态与脉冲执行。目标样本必须新鲜、连续且满足窗口，丢失后停止，不能拿上一帧结果继续驱动。

配置是 `src/aic_competition/config/competition.yaml`。其中航点是遗留示例，不是你的地图坐标；所有未确认标定仍为false。不能为了让程序启动把这些标记批量改为true。

离线入口：

```bash
python simulate_ros.py --scenario success
python simulate_ros.py --scenario target_lost
```

第二条预期失败，用于查看故障收尾。仿真不启动ROS。若需要ROS节点，先在ASCII路径下建立catkin工作空间，并准备对应Python版本的ROS依赖。在工作空间根目录构建后：

```bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
roslaunch aic_competition competition.launch
rosservice call /competition/start
```

这里只启动默认 `dry_run=true` 的上层仿真节点，不启动厂家底盘、导航和真实发射。Melodic的默认Python环境不一定能直接运行Python3节点；不要仅改shebang或强行指定解释器就认定兼容。

真实运行还需合法取得底盘驱动、配置导航速度输出到安全桥输入、接入视觉和播音服务。接口见资源说明。发布包不包含厂家全栈启动文件，不提供不经标定的实车“一键跑”命令。

## 2. 定时路线模式

现场链路是：起点右移、前进、转向 → 停车取新图 → 视觉结果校验 → 播放本地答案 → 转回 → 1号位置脉冲及后退 → 2号位置脉冲及后退 → 3号位置脉冲及后退 → 末尾右移 → 本地结束播报。

参数集中在 `config/timed_route.json`：

| 参数 | 含义 |
|---|---|
| `steps` | 起点两段平移和转向 |
| `speed_m_s` / `turn_speed_rad_s` | 时间换算所用速度 |
| `after_return` | 1号位置前进、回退 |
| `after_retreat` | 2/3号接续和末尾右移 |
| `shots_per_position` | 每个位置的软件脉冲次数，示例为2 |

距离保留自现场工程快照，只用于理解参数组织，不是通用路线。配置的0.72米前进或1.31米平移，不意味着另一辆车也能走到同一位置。轮胎、地面、电量、负载、速度建立时间和底盘坐标方向都会影响结果。

不带 `--live` 就不会执行真实运动。`--yes` 只省去确认词，不绕过设备缺失、占用、参数或前序故障。起点接续模式没有额外路线核验勾选门槛；这不等于具有避障能力或路线已经安全。

原车接口使用底盘 `/dev/abotbase` 与发射器 `/dev/shoot`。它们是设备别名，不是Linux必然提供的文件；其他平台必须先适配协议。缺设备时不要猜 `ttyUSB0`，先用只读诊断收集身份：

```bash
python src/aic_competition/tools/manual_shoot.py --diagnose
```

此操作会在本地 `runs/` 保存设备和历史执行信息。报告可能包含USB序列号、进程或路径信息，不要未经检查就上传。
