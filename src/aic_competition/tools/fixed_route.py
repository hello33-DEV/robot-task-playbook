#!/usr/bin/env python3
"""Fixed timed motion; optional card recognition after start-pose motion. Preview by default."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import sys
import time
import uuid


sys.path.insert(0, str(Path(__file__).resolve().parent))
import manual_drive as drive
import card_camera as card
import manual_shoot as shoot


PROJECT = Path(__file__).resolve().parents[3]
ACTIONS = {"forward": (1, 0, 0), "backward": (-1, 0, 0),
           "strafe_left": (0, 1, 0), "strafe_right": (0, -1, 0),
           "left": (0, 0, 1), "right": (0, 0, -1), "stop": (0, 0, 0)}
SETTLE_SECONDS = 0.25
LIVE_STEP_LIMIT = 0.50
ROUTE_CHECKS = ("layout_and_start_pose", "motion_timings", "physical_stop_and_watchdog",
                "individual_steps_tested")
ROUTE_CHECKPOINTS = ("识别位", "1号任务位", "2号任务位", "3号任务位", "终点")
START_POSE_TEST_KIND = "start_pose_motion_test"
START_POSE_TEST_ACTIONS = ("strafe_right", "forward", "left")


def number(value, maximum, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label + "必须填写数值")
    return drive.bounded_number(value, maximum, label)


def load_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(plan, dict) or type(plan.get("schema_version")) is not int or plan["schema_version"] != 1:
        raise ValueError("路线schema_version必须为1")
    number(plan.get("speed_m_s"), drive.MAX_LINEAR_SPEED, "线速度")
    number(plan.get("turn_speed_rad_s"), drive.MAX_ANGULAR_SPEED, "角速度")
    shots_per_position = plan.get("shots_per_position", 1)
    if isinstance(shots_per_position, bool) or not isinstance(shots_per_position, int) \
            or not 1 <= shots_per_position <= 10:
        raise ValueError("shots_per_position必须是1至10的整数")
    heading = plan.get("initial_heading")
    if heading is not None and (not isinstance(heading, str) or not heading.strip()):
        raise ValueError("initial_heading应为朝向说明或null")
    checks = plan.get("route_checks", {})
    if not isinstance(checks, dict) or any(type(value) is not bool for value in checks.values()):
        raise ValueError("route_checks必须使用布尔值，不能用字符串或数字代替")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 100:
        raise ValueError("路线必须包含1至100个动作")
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("name"), str) or not step["name"].strip():
            raise ValueError("每个动作必须有名称")
        action = step.get("action")
        if action is not None and (not isinstance(action, str) or action not in ACTIONS):
            raise ValueError("action仅允许forward/backward/strafe_left/strafe_right/left/right/stop或null")
        if step.get("seconds") is not None:
            number(step["seconds"], 60.0, "单段秒数")
        checkpoint = step.get("checkpoint")
        if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()
                                       or action != "stop"):
            raise ValueError("checkpoint只能填在stop停车动作中，且必须为非空名称")
    return plan


def prepare_start_pose_test(plan):
    if plan.get("test_kind") != START_POSE_TEST_KIND:
        raise ValueError("--start-pose-test只能使用test_kind为start_pose_motion_test的专用配置")
    if len(plan["steps"]) != 3:
        raise ValueError("起点姿态运动测试必须恰好包含3个动作")
    if tuple(step.get("action") for step in plan["steps"]) != START_POSE_TEST_ACTIONS:
        raise ValueError("起点姿态运动测试动作必须依次为右移、前进、逆时针旋转")
    prepared = copy.deepcopy(plan)
    for step in prepared["steps"]:
        action = step["action"]
        if action in ("strafe_right", "forward"):
            distance = number(step.get("target_distance_m"), 2.0, "目标距离")
            step["seconds"] = distance / plan["speed_m_s"]
        else:
            angle = number(step.get("target_angle_deg"), 360.0, "目标角度")
            step["seconds"] = math.radians(angle) / plan["turn_speed_rad_s"]
        if step["seconds"] > 60.0:
            raise ValueError("起点姿态测试单段开环时间超过60秒；先降低目标或重新标定速度")
    return prepared


def demo_plan():
    return {"schema_version": 1, "initial_heading": "离线演示，不代表现场朝向",
            "speed_m_s": 0.05, "turn_speed_rad_s": 0.20,
            "steps": [{"name": "演示：" + action, "action": action, "seconds": 0.25}
                      for action in ("forward", "backward", "left", "right")]}


def require_route_checks(plan, full_route=False, continuation=False):
    checks = plan.get("route_checks", {})
    required = ROUTE_CHECKS if full_route or continuation else ROUTE_CHECKS[:3]
    missing = [name for name in required if checks.get(name) is not True]
    if missing:
        raise ValueError("路线尚未完成现场核验：" + ", ".join(missing) + "；不得为运行而直接改true")
    if full_route:
        checkpoints = tuple(step["checkpoint"] for step in plan["steps"] if step.get("checkpoint"))
        if checkpoints != ROUTE_CHECKPOINTS or plan["steps"][-1].get("checkpoint") != "终点":
            raise ValueError("整场必须依次包含识别位、1/2/3号任务位和终点停车，且终点停车必须为最后动作")


def select_steps(plan, index=None):
    if index is not None and not 1 <= index <= len(plan["steps"]):
        raise ValueError("--step超出动作序号范围（从1开始）")
    chosen = list(enumerate(plan["steps"], 1))
    if index is not None:
        chosen = [chosen[index - 1]]
    if not plan.get("initial_heading"):
        raise ValueError("尚未填写initial_heading；先明确并固定起始朝向")
    for ordinal, step in chosen:
        if step.get("action") is None or step.get("seconds") is None:
            raise ValueError("第{}段未配置动作或时长；不使用猜测参数运行".format(ordinal))
    if sum(step["seconds"] + SETTLE_SECONDS for _, step in chosen) > 120.0:
        raise ValueError("所选动作及段间零速保持时间超过120秒软件预算")
    return chosen


def velocity(plan, step):
    axes = ACTIONS[step["action"]]
    return (axes[0] * plan["speed_m_s"], axes[1] * plan["speed_m_s"],
            axes[2] * plan["turn_speed_rad_s"])


class VirtualClock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def hold(transport, command, seconds, clock, sleep):
    previous = clock()
    deadline = previous + seconds
    while True:
        now = clock()
        if now < previous or now - previous > 0.15:
            raise RuntimeError("运动循环延迟或时钟异常；中止，不继续下一段")
        previous = now
        if now >= deadline:
            return
        transport.send(command)
        remaining = deadline - clock()
        if remaining > 0:
            sleep(min(drive.TICK, remaining))


def execute(plan, chosen, transport, events, clock=time.monotonic, sleep=time.sleep):
    try:
        transport.send(drive.ZERO)
        for ordinal, step in chosen:
            print("第{}段：{} / {} / {:.3f}秒".format(
                ordinal, step["name"], step["action"], step["seconds"]), flush=True)
            event = {"step": ordinal, "name": step["name"], "action": step["action"],
                     "requested_seconds": step["seconds"], "software_step_complete": False}
            if step.get("checkpoint"):
                event["checkpoint"] = step["checkpoint"]
            events.append(event)
            hold(transport, velocity(plan, step), step["seconds"], clock, sleep)
            transport.send(drive.ZERO)
            hold(transport, drive.ZERO, SETTLE_SECONDS, clock, sleep)
            event["software_step_complete"] = True
            print("本段软件发送结束；已请求零速，不代表已到位或物理停车确认", flush=True)
            if step.get("checkpoint"):
                print("计划点位[{}]驻留发送结束；没有定位反馈，不能据此认定到点".format(step["checkpoint"]), flush=True)
    finally:
        transport.send(drive.ZERO)


def save_result(path, result):
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_post_return_steps(plan):
    settings = plan.get("after_return")
    if not isinstance(settings, dict):
        raise ValueError("发射后退流程必须配置after_return中的前进和后退目标距离")
    steps = []
    for ordinal, action, key, name in ((5, "forward", "approach_distance_m", "回转后前进"),
                                        (6, "backward", "retreat_distance_m", "发射后后退")):
        distance = number(settings.get(key), 2.0, name + "距离")
        seconds = distance / plan["speed_m_s"]
        if seconds > 60.0:
            raise ValueError(name + "单段开环时间超过60秒")
        steps.append((ordinal, {"name": name + "{}厘米目标".format(distance * 100),
                               "action": action, "target_distance_m": distance, "seconds": seconds}))
    return steps


def prepare_continuation_steps(plan):
    settings = plan.get("after_retreat")
    if not isinstance(settings, dict):
        raise ValueError("二三号位接续路线必须配置after_retreat六段目标距离")
    # 地图对应：从1号回退回公共横向通道，依次横移至2/3号区域，再前进到理论射击位。
    # 图示只说明拓扑关系；距离仍是JSON中的开环目标，不代表实测到位。
    stages = ((7, "strafe_right", "to_second_strafe_m", "1号位回退后沿公共通道右移至2号区域"),
              (8, "forward", "to_second_approach_m", "从通道前进至2号理论射击位（不识别/瞄准）"),
              (9, "backward", "from_second_retreat_m", "2号射击流程后退回公共通道"),
              (10, "strafe_right", "to_third_strafe_m", "沿公共通道右移至3号区域"),
              (11, "forward", "to_third_approach_m", "从通道前进至3号理论射击位（不识别/瞄准）"),
              (12, "backward", "from_third_retreat_m", "3号射击流程后退离开任务位"),
              (13, "strafe_right", "return_to_finish_strafe_m", "最后右移，作为继续返回终点方向的接续短移"))
    steps = []
    for ordinal, action, key, name in stages:
        distance = number(settings.get(key), 2.0, name + "距离")
        seconds = distance / plan["speed_m_s"]
        if seconds > 60.0:
            raise ValueError(name + "单段开环时间超过60秒")
        steps.append((ordinal, {"name": name + "{}厘米目标".format(distance * 100),
                                "action": action, "target_distance_m": distance, "seconds": seconds}))
    return steps


def shot_succeeded(shot):
    return (shot["request_accepted"] is True
            and shot["software_execution_complete"] is True
            and shot["pulse_result"] == "completed"
            and not shot.get("fault")
            and not shot.get("worker_still_running", True))


def refresh_shot_summary(batch):
    records = batch["shot_records"]
    batch["shots_completed"] = sum(shot_succeeded(shot) for shot in records)
    batch["request_accepted"] = bool(records) and all(shot["request_accepted"] is True for shot in records)
    batch["software_execution_complete"] = bool(records) and all(
        shot["software_execution_complete"] is True for shot in records)
    batch["pulse_result"] = "completed" if batch["software_execution_complete"] else (
        records[-1].get("pulse_result", "not_requested") if records else "not_requested")
    batch["fault"] = next((shot.get("fault") for shot in records if shot.get("fault")), None)
    batch["worker_still_running"] = any(shot.get("worker_still_running", True) for shot in records)


def execute_shot_batch(batch, target_label, live, save):
    shot_count = batch["shots_requested"]
    for shot_number in range(1, shot_count + 1):
        shot = shoot.new_record(live)
        shot["target_label"] = target_label
        shot["shot_number"] = shot_number
        shot["shots_requested"] = shot_count
        batch["shot_records"].append(shot)
        refresh_shot_summary(batch)
        save()
        transport = None
        try:
            transport = shoot.open_serial() if live else None
            shoot.execute_once(transport, shot, save)
        except BaseException:
            refresh_shot_summary(batch)
            save()
            raise
        finally:
            if transport is not None:
                try:
                    transport.close()
                except BaseException as error:
                    shot["fault"] = shot.get("fault") or "发射串口关闭失败：{}: {}".format(
                        type(error).__name__, error)
                    refresh_shot_summary(batch)
                    save()
                    raise
        refresh_shot_summary(batch)
        save()
        if not shot_succeeded(shot):
            raise RuntimeError("{}号第{}次发射脉冲或停止清理未确认软件结束，禁止继续移动；不自动补射".format(
                target_label, shot_number))
        print("{}号位置第{}次脉冲与停止清理软件结束；未确认实际出弹或命中。".format(
            target_label, shot_number), flush=True)
        if shot_number < shot_count:
            time.sleep(shoot.CONFIG["cooldown_seconds"])


def execute_target_shot(result, target, result_path):
    shot = result["target_shots"][str(target)]
    shot["target_label"] = target
    save_result(result_path, result)
    print("到达{}号理论射击位置（仅按开环目标）；不识别、不瞄准，执行{}次0.10秒脉冲。".format(
        target, shot["shots_requested"]), flush=True)
    execute_shot_batch(shot, target, True, lambda: save_result(result_path, result))
    shot["status"] = "software_complete"
    save_result(result_path, result)
    time.sleep(shoot.CONFIG["cooldown_seconds"])
    print("{}号共{}次脉冲软件结束；未确认实际出弹或命中。".format(
        target, shot["shots_completed"]), flush=True)


def broadcast_finish(result, result_path, live):
    result["finish_broadcast_text"] = card.FINISH_TEXT
    if not live:
        result["finish_audio_file"] = None
        result["finish_broadcast_status"] = "simulation_skipped"
        save_result(result_path, result)
        print("模拟：跳过本地终点播报，不读取音频、不播放；未确认实际到达终点。", flush=True)
        return
    audio_path = card.finish_audio()
    result["finish_audio_file"] = str(audio_path)
    result["finish_broadcast_status"] = "running"
    save_result(result_path, result)
    print("已请求零速；开始播放本地终点语音：{}。".format(audio_path.name), flush=True)
    try:
        card.play_audio(audio_path)
    except BaseException as error:
        result["finish_broadcast_status"] = "failed"
        result["finish_broadcast_error"] = str(error)[:2000]
        result["workflow_software_complete"] = False
        save_result(result_path, result)
        raise
    result["finish_broadcast_status"] = "player_returned_success"
    save_result(result_path, result)
    print("终点播放器已返回；仍需人工确认实际听到语音。", flush=True)


def execute_shoot_and_retreat(plan, steps, port, result, result_path, continue_requested=False):
    result["approach_status"] = "running"
    save_result(result_path, result)
    drive.check_port(port)
    with drive.SerialPort(port) as transport:
        execute(plan, [steps[0]], transport, result["approach_events"], time.monotonic, time.sleep)
    result["approach_status"] = "software_complete"
    result["launch_status"] = "running"
    save_result(result_path, result)
    print("前进及零速清理软件结束；不识别靶子，执行{}次0.10秒玩具发射脉冲。".format(
        result["launch_result"]["shots_requested"]), flush=True)
    execute_shot_batch(result["launch_result"], 1, True,
                       lambda: save_result(result_path, result))
    launched = result["launch_result"]
    launched["status"] = "software_complete"
    result["launch_status"] = "pulse_software_complete"
    save_result(result_path, result)
    time.sleep(shoot.CONFIG["cooldown_seconds"])
    print("脉冲与停止清理软件结束；未确认实际出弹或命中，开始后退目标。", flush=True)
    result["retreat_status"] = "running"
    save_result(result_path, result)
    drive.check_port(port)
    with drive.SerialPort(port) as transport:
        execute(plan, [steps[1]], transport, result["retreat_events"], time.monotonic, time.sleep)
    result["retreat_status"] = "software_complete"
    result["workflow_software_complete"] = not continue_requested
    result["status"] = "motion_recognition_return_launch_and_retreat_software_complete"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="内置虚拟时间演示，不接设备")
    mode.add_argument("--demo-route", action="store_true", help="整场动作离线演示，运动时长统一替换为虚拟0.25秒")
    mode.add_argument("--simulate", action="store_true", help="模拟已填写的路线，不接设备")
    mode.add_argument("--check", action="store_true", help="只检查底盘设备，不打开串口")
    mode.add_argument("--live", action="store_true", help="真实运动，须--step或经核验的--full-route")
    parser.add_argument("--full-route", action="store_true", help="配合--live，核验完整路线后一次执行全部动作")
    parser.add_argument("--start-pose-test", action="store_true",
                        help="执行专用三动作测试，距离与角度读取起点配置")
    parser.add_argument("--recognize-after-stop", action="store_true",
                        help="仅用于起点三动作：停车后实拍识别、本地播报成功再顺时针转回；真实模式调用付费API")
    parser.add_argument("--shoot-and-retreat", action="store_true",
                        help="须配合起点识别联动：回转后按配置前进、执行配置次数的玩具发射脉冲、后退；真实模式输入RUN TEST FIRE，或显式--yes免确认词")
    parser.add_argument("--continue-to-3", action="store_true",
                        help="显式选择1号回退后的接续路线；2/3号各执行配置次数的发射脉冲，不识别、不瞄准")
    parser.add_argument("--yes", action="store_true",
                        help="仅用于真实起点识别及发射后退全流程：命令本身确认运动/API/发射，不再输入RUN TEST FIRE；保留现场终端和设备预检")
    parser.add_argument("--camera", help="识别须显式指定已核对的彩色相机编号、/dev/video设备或openni2")
    parser.add_argument("--vision-timeout", type=float, default=30.0,
                        help="每次视觉网络请求的超时秒数，不是全流程总时限")
    parser.add_argument("--plan", type=Path, default=PROJECT / "config/route_template.json")
    parser.add_argument("--step", type=int, help="选择第几个动作，从1开始")
    parser.add_argument("--port", default="/dev/abotbase")
    args = parser.parse_args(argv)
    result = None
    result_path = None
    exit_code = 0
    try:
        if args.yes and not (args.live and args.start_pose_test and args.recognize_after_stop
                             and args.shoot_and_retreat):
            raise ValueError("--yes只用于--live --start-pose-test --recognize-after-stop --shoot-and-retreat完整联动")
        if args.shoot_and_retreat and not args.recognize_after_stop:
            raise ValueError("--shoot-and-retreat必须配合--recognize-after-stop，不单独运行或跳过前序流程")
        if args.continue_to_3 and not args.shoot_and_retreat:
            raise ValueError("--continue-to-3必须配合--shoot-and-retreat，在1号回退完成后接续")
        if args.shoot_and_retreat and Path(args.port).resolve() == Path(shoot.PORT).resolve():
            raise ValueError("底盘与发射器不能使用同一串口")
        if args.recognize_after_stop:
            if not args.start_pose_test or not (args.live or args.simulate):
                raise ValueError("--recognize-after-stop只支持--start-pose-test配合--live或--simulate")
            if not args.camera or not args.camera.strip():
                raise ValueError("停车后识别须明确指定--camera，不自动猜相机或使用旧照片")
            number(args.vision_timeout, 120.0, "视觉请求超时")
        elif args.camera is not None:
            raise ValueError("--camera须配合--recognize-after-stop")
        if args.full_route and (not args.live or args.step is not None):
            raise ValueError("--full-route只能配合--live，不能同时使用--step或演示/模拟模式")
        if args.start_pose_test and (args.full_route or args.step is not None or args.demo or args.demo_route or args.check):
            raise ValueError("--start-pose-test不能与--full-route、--step、--demo、--demo-route或--check同时使用")
        if args.start_pose_test and not (args.live or args.simulate):
            raise ValueError("--start-pose-test必须配合--live或--simulate")
        if args.check:
            drive.check_port(args.port)
            print("底盘设备路径/权限/占用检查通过；未开串口，未确认实际运动或停车")
            return 0
        plan = demo_plan() if args.demo else load_plan(args.plan)
        if args.start_pose_test:
            plan = prepare_start_pose_test(plan)
        if args.demo_route:
            print("整场虚拟演示：所有运动时长替换为0.25秒，不代表现场距离或90度转角，不开设备。")
            plan["initial_heading"] = "虚拟假设车头朝场地图上方，非现场确认"
            for step in plan["steps"]:
                if step["action"] != "stop":
                    step["seconds"] = 0.25
        if not (args.demo or args.demo_route or args.simulate or args.live):
            print("仅预览，不连接小车。修改参数前先阅读docs/MODES.md。")
            print("起始朝向：{}；线速={}m/s，角速={}rad/s".format(
                plan.get("initial_heading") or "待确定", plan["speed_m_s"], plan["turn_speed_rad_s"]))
            for ordinal, step in enumerate(plan["steps"], 1):
                print("{}：{} / {} / {}秒".format(ordinal, step["name"],
                      step.get("action") or "待实测", step.get("seconds") if step.get("seconds") is not None else "待实测"))
            return 0
        chosen = select_steps(plan, args.step)
        return_step = None
        post_return_steps = prepare_post_return_steps(plan) if args.shoot_and_retreat else []
        continuation_steps = prepare_continuation_steps(plan) if args.continue_to_3 else []
        if args.recognize_after_stop:
            return_step = {"name": "播报后顺时针回转{}度目标".format(plan["steps"][2]["target_angle_deg"]),
                           "action": "right", "target_angle_deg": plan["steps"][2]["target_angle_deg"],
                           "seconds": plan["steps"][2]["seconds"]}
            if sum(step["seconds"] + SETTLE_SECONDS
                   for _, step in chosen + [(4, return_step)] + post_return_steps + continuation_steps) > 120.0:
                raise ValueError("含回转、发射前后及接续运动的段间零速保持超过120秒软件运动预算（不含识别和脉冲）")
        if args.live:
            if args.full_route:
                require_route_checks(plan, full_route=True)
            elif not args.start_pose_test:
                if args.step is None:
                    raise ValueError("--live必须指定--step，或用--full-route执行完整路线")
                if chosen[0][1]["seconds"] > LIVE_STEP_LIMIT:
                    require_route_checks(plan)
            if not sys.stdin.isatty():
                raise RuntimeError("真实运动要求现场交互终端，不接受管道确认")
            drive.check_port(args.port)
            if args.shoot_and_retreat:
                shoot.check_live()
            if args.recognize_after_stop:
                card.prepare_recognition(args.camera)
        stamp = datetime.now(timezone.utc)
        result_path = PROJECT / "runs" / "固定路线" / (stamp.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]) / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result = {"created_at": stamp.isoformat(), "mode": ("live_start_pose_test" if args.start_pose_test else ("live_full_route" if args.full_route else "live_single_step")) if args.live else ("start_pose_test_simulation" if args.start_pose_test else "simulation"),
                  "plan": "built_in_demo" if args.demo else str(args.plan.resolve()),
                  "virtual_route_demo": args.demo_route,
                  "confirmation_mode": "explicit_command_flag" if args.yes else "interactive" if args.live else "not_required_simulation",
                  "requested_steps": [{"step": ordinal, **step} for ordinal, step in chosen],
                  "route_checks": plan.get("route_checks", {}),
                  "settings": {"initial_heading": plan["initial_heading"],
                               "speed_m_s": plan["speed_m_s"], "turn_speed_rad_s": plan["turn_speed_rad_s"],
                               "shots_per_position": plan.get("shots_per_position", 1),
                               "recognition_camera": args.camera,
                               "vision_timeout_s": args.vision_timeout if args.recognize_after_stop else None},
                  "status": "prepared", "software_sequence_complete": False,
                  "recognition_requested": args.recognize_after_stop,
                  "recognition_status": "not_started",
                  "return_heading_requested": args.recognize_after_stop,
                  "requested_return_step": return_step,
                  "return_heading_status": "not_started", "return_events": [],
                  "shoot_and_retreat_requested": args.shoot_and_retreat,
                  "requested_post_return_steps": [{"step": ordinal, **step} for ordinal, step in post_return_steps],
                  "approach_status": "not_started", "approach_events": [],
                  "launch_status": "not_started", "launch_result": (
                      dict(shoot.new_record(args.live), shots_requested=plan.get("shots_per_position", 1),
                           shots_completed=0, shot_records=[])
                      if args.shoot_and_retreat else None),
                  "retreat_status": "not_started", "retreat_events": [],
                  "continuation_requested": args.continue_to_3,
                  "requested_continuation_steps": [{"step": ordinal, **step} for ordinal, step in continuation_steps],
                  "target_shots": ({str(target): dict(shoot.new_record(args.live),
                                                       shots_requested=plan.get("shots_per_position", 1),
                                                       shots_completed=0, shot_records=[])
                                    for target in (2, 3)}
                                   if args.continue_to_3 else {}),
                  "continuation_status": "not_started", "continuation_events": [],
                  "workflow_software_complete": False, "physical_heading_confirmed": False,
                  "physical_audio_confirmed": False,
                  "physical_motion_confirmed": False, "physical_distance_confirmed": False,
                  "physical_stop_confirmed": False, "physical_route_confirmed": False, "events": []}
        save_result(result_path, result)
        if args.live:
            if args.start_pose_test:
                print("真实起点姿态三动作测试：按下列配置顺序执行。")
                print("时间按标称速度开环计算：不等于实际距离/角度；右移依赖全向或麦克纳姆底盘。")
                for ordinal, step in chosen:
                    target = ("{}m目标".format(step["target_distance_m"])
                              if step["action"] in ("strafe_right", "forward")
                              else "{}°目标".format(step["target_angle_deg"]))
                    print("{}：{} / {} / {} / {:.3f}秒".format(
                        ordinal, step["name"], step["action"], target, step["seconds"]))
            elif args.full_route:
                print("真实整场：共{}个动作，一次确认后自动顺序执行，不再逐段等输入。".format(len(chosen)))
                for ordinal, step in chosen:
                    print("{}：{} / {} / {}秒".format(ordinal, step["name"], step["action"], step["seconds"]))
            else:
                print("真实单段：{} / {} / {}秒".format(chosen[0][1]["name"], chosen[0][1]["action"], chosen[0][1]["seconds"]))
            if args.shoot_and_retreat:
                print("本次包含玩具发射：确认朝向无人靶区、全员护目镜、有可靠底盘急停及发射器断电手段。")
                print("停止其他发射和底盘程序；软件不确认物理停车、实际出弹、出弹数量或命中。")
                print("回转后前进{}厘米目标（{:.3f}秒）→{shots}次0.10秒脉冲（每次停止清理及0.45秒冷却）→后退{}厘米目标（{:.3f}秒）。".format(
                    post_return_steps[0][1]["target_distance_m"] * 100, post_return_steps[0][1]["seconds"],
                    post_return_steps[1][1]["target_distance_m"] * 100, post_return_steps[1][1]["seconds"],
                    shots=plan.get("shots_per_position", 1)))
                if args.continue_to_3:
                    print("1号回退后按JSON距离沿地图通道到2/3号理论位置，各到位后按配置执行{}次脉冲；不识别、不瞄准。".format(plan.get("shots_per_position", 1)))
            else:
                print("确认安全地面、发射器断能、无人靠近、有可靠物理停车手段；其他底盘程序已关闭。")
            if args.recognize_after_stop:
                print("本次确认也授权实拍API：首张模型自评可靠可直播，否则最多3张两次一致；超时最后加拍1次。")
                print("播报成功后自动顺时针回转{}°目标，约{:.3f}秒；回转失败不自动重试。".format(
                    return_step["target_angle_deg"], return_step["seconds"]))
                print("全程无传感器确认实际角度、停车或恢复朝向；异常须现场中止，回转不是闭环校正。")
            confirmation = "RUN TEST" if args.start_pose_test else ("RUN ROUTE" if args.full_route else "MOVE")
            if args.shoot_and_retreat:
                confirmation = "RUN TEST FIRE"
            if args.yes:
                print("已通过--yes明确确认整条真实运动、付费识别及各射击位按配置执行{}次玩具发射；不再等待确认词，预检和故障中止仍生效。".format(plan.get("shots_per_position", 1)), flush=True)
            elif input("确认起始朝向和上述条件后输入 {}（否则取消）：".format(confirmation)).strip() != confirmation:
                result["status"] = "cancelled"
                print("已取消，未打开串口")
        if result["status"] != "cancelled":
            previous_handler = signal.signal(signal.SIGTERM, drive.handle_termination)
            try:
                result["status"] = "running"
                save_result(result_path, result)
                virtual = VirtualClock()
                with (drive.SerialPort(args.port) if args.live else drive.Simulation()) as transport:
                    execute(plan, chosen, transport, result["events"],
                            time.monotonic if args.live else virtual.now,
                            time.sleep if args.live else virtual.sleep)
                result["software_sequence_complete"] = True
                result["status"] = "software_sequence_complete" if args.live else "simulation_complete"
                result["workflow_software_complete"] = not args.recognize_after_stop
                if args.full_route:
                    broadcast_finish(result, result_path, args.live)
                if args.recognize_after_stop:
                    if args.live:
                        session = result_path.parent / "recognition"
                        result["recognition_session"] = str(session)
                        result["recognition_status"] = "running"
                        save_result(result_path, result)
                        print("三动作软件发送结束，零速驻留后串口已关闭；开始新实拍识别与本地播报。", flush=True)
                        result["recognition_result"] = card.run_auto(args.camera, session, args.vision_timeout)
                        result["recognition_status"] = "recognition_and_playback_software_complete"
                        result["return_heading_status"] = "running"
                        save_result(result_path, result)
                        print("识别与播放器已正常返回；重新检查底盘占用，开始顺时针回转目标。", flush=True)
                        drive.check_port(args.port)
                        with drive.SerialPort(args.port) as transport:
                            execute(plan, [(4, return_step)], transport, result["return_events"],
                                    time.monotonic, time.sleep)
                        result["return_heading_status"] = "software_complete"
                        result["workflow_software_complete"] = not args.shoot_and_retreat
                        result["status"] = "motion_recognition_and_return_software_complete"
                        if args.shoot_and_retreat:
                            execute_shoot_and_retreat(plan, post_return_steps, args.port, result, result_path, args.continue_to_3)
                            if args.continue_to_3:
                                result["continuation_status"] = "running"
                                save_result(result_path, result)
                                drive.check_port(args.port)
                                for ordinal, step in continuation_steps:
                                    drive.check_port(args.port)
                                    with drive.SerialPort(args.port) as transport:
                                        execute(plan, [(ordinal, step)], transport, result["continuation_events"],
                                                time.monotonic, time.sleep)
                                    if ordinal == 8:
                                        execute_target_shot(result, 2, result_path)
                                    elif ordinal == 11:
                                        execute_target_shot(result, 3, result_path)
                                result["continuation_status"] = "software_complete"
                                result["workflow_software_complete"] = True
                                result["status"] = "motion_recognition_launch_retreat_and_continuation_software_complete"
                                broadcast_finish(result, result_path, args.live)
                    else:
                        result["recognition_status"] = "simulation_skipped"
                        result["return_heading_status"] = "simulation_skipped"
                        print("模拟：此处识别播报；成功后才顺时针回转{}°目标（{:.3f}秒）。".format(
                            return_step["target_angle_deg"], return_step["seconds"]))
                        print("不开相机、不请求API、不播放、不伪造答案；跳过依赖真实播报成功的回转。")
                        if args.shoot_and_retreat:
                            for stage in ("approach_status", "launch_status", "retreat_status"):
                                result[stage] = "simulation_skipped"
                            print("模拟：回转成功后前进{}厘米目标（{:.3f}秒）→1号位{}次0.10秒发射脉冲→后退{}厘米目标（{:.3f}秒）；全部条件跳过，不连接设备。".format(
                                post_return_steps[0][1]["target_distance_m"] * 100, post_return_steps[0][1]["seconds"],
                                result["launch_result"]["shots_requested"],
                                post_return_steps[1][1]["target_distance_m"] * 100, post_return_steps[1][1]["seconds"]))
                            if args.continue_to_3:
                                result["continuation_status"] = "simulation_skipped"
                                for shot in result["target_shots"].values():
                                    shot["status"] = "simulation_skipped"
                                    shot["pulse_result"] = "not_requested"
                                for ordinal, step in continuation_steps:
                                    print("模拟条件跳过：第{}段 {} / {} / {:.3f}秒；不连接设备、不发射。".format(
                                        ordinal, step["name"], step["action"], step["seconds"]))
                                print("模拟：2号、3号位各执行{}次0.10秒脉冲；本次均跳过，不连接设备。".format(
                                    plan.get("shots_per_position", 1)))
                                broadcast_finish(result, result_path, args.live)
            finally:
                signal.signal(signal.SIGTERM, previous_handler)
            print("软件流程结束；没有反馈测距或到点确认，不代表整场路线通过。")
    except KeyboardInterrupt:
        exit_code = 130
        if result is not None:
            result["status"] = "interrupted"
            if result["recognition_status"] == "running":
                result["recognition_status"] = "interrupted"
            if result["return_heading_status"] == "running":
                result["return_heading_status"] = "interrupted"
            for stage in ("approach_status", "launch_status", "retreat_status", "continuation_status"):
                if result[stage] == "running":
                    result[stage] = "interrupted"
                    if stage == "launch_status":
                        result["launch_result"]["status"] = "interrupted"
        print("已中断并尝试零速；须现场确认停车", file=sys.stderr)
    except Exception as error:
        exit_code = 2
        if result is not None:
            result["status"] = "failed"
            if result["recognition_status"] == "running":
                result["recognition_status"] = "failed"
            if result["return_heading_status"] == "running":
                result["return_heading_status"] = "failed"
            for stage in ("approach_status", "launch_status", "retreat_status", "continuation_status"):
                if result[stage] == "running":
                    result[stage] = "failed"
                    if stage == "launch_status":
                        result["launch_result"]["status"] = "failed"
            result["error"] = str(error)
            result["error_type"] = type(error).__name__
        print("固定路线失败：{}；不据此认定已停车或到位".format(error), file=sys.stderr)
    finally:
        if result is not None and result_path is not None:
            result["checkpoints_software_complete"] = [event["checkpoint"] for event in result["events"]
                                                      if event.get("checkpoint") and event["software_step_complete"]]
            try:
                save_result(result_path, result)
                print("记录：" + str(result_path))
            except OSError as error:
                exit_code = 2
                print("结果记录失败：{}；如曾实跑须现场确认停车".format(error), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
