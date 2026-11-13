#!/usr/bin/env python3
"""Standalone, manually confirmed toy-launcher pulse; disconnected simulation by default."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid


PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE.parent.parent
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(PACKAGE / "tools"))
from aic_competition.pulse import PulseDriver
from manual_drive import check_port, handle_termination, port_users


PORT = "/dev/shoot"
CONFIG = dict(start_hex="55 01 12 00 00 00 01 69",
              stop_hex="55 01 11 00 00 00 01 68",
              pulse_seconds=0.10, cooldown_seconds=0.45, write_timeout=0.10)


def check_live():
    check_port(PORT, label="发射器")
    if importlib.util.find_spec("serial") is None:
        raise RuntimeError("当前Python3缺少pyserial；未打开串口，请先完成环境检查")


def open_serial():
    import fcntl
    import termios
    import serial
    check_live()
    transport = serial.Serial(PORT, 9600, bytesize=8, parity="N", stopbits=1,
                              timeout=0.05, write_timeout=CONFIG["write_timeout"],
                              exclusive=True)
    try:
        fcntl.ioctl(transport.fileno(), termios.TIOCEXCL)
        others = port_users(PORT) - {os.getpid()}
        if others:
            raise RuntimeError("串口出现并发占用，拒绝发送：{}".format(sorted(others)))
        return transport
    except BaseException:
        transport.close()
        raise


class LoggedSerial:
    def __init__(self, transport, record, announce=True):
        self.transport = transport
        self.record = record
        self.announce = announce

    def write(self, frame):
        entry = dict(command="START" if frame == bytes.fromhex(CONFIG["start_hex"]) else "STOP",
                     frame_hex=frame.hex().upper(), bytes_requested=len(frame), bytes_written=None,
                     status="attempted", timestamp=datetime.now(timezone.utc).isoformat())
        self.record["serial_writes"].append(entry)
        try:
            written = self.transport.write(frame)
        except Exception as error:
            entry.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        entry.update(bytes_written=written, status="written" if written == len(frame) else "short_write")
        if self.announce:
            print("发射串口 {}：写入{}/{}字节；不是设备回执或实际出弹确认。".format(
                entry["command"], written, len(frame)), flush=True)
        return written

    def close(self):
        self.transport.close()


def execute_once(transport, record, save=lambda: None):
    driver = PulseDriver(CONFIG, LoggedSerial(transport, record) if transport is not None else None)
    try:
        driver.stop()
        if driver.snapshot()["fault"]:
            raise RuntimeError("启动STOP写入失败，禁止发送START")
        driver.arm(True)
        record["events"].append("PULSE_REQUEST_ATTEMPT")
        save()
        accepted = driver.request(lambda: True)
        record["request_accepted"] = accepted
        if not accepted:
            raise RuntimeError("请求未被接受；不重试")
        record["pulse_id"] = driver.snapshot()["pulse_id"]
        record["events"].append("PULSE_REQUEST_ACCEPTED")
        save()
        print("请求已接受；这不表示执行结束或实际出弹。")
        driver.thread.join(timeout=CONFIG["pulse_seconds"] + CONFIG["write_timeout"] * 3 + 0.5)
        state = driver.snapshot()
        if driver.thread.is_alive() or state["busy"] or state["pulse_result"] != "completed" or state["fault"]:
            raise RuntimeError("执行未正常结束：{}；不重试，现场确认机构是否停止".format(state))
    finally:
        driver.close()
        state = driver.snapshot()
        record["pulse_result"] = state["pulse_result"]
        record["fault"] = state["fault"]
        record["worker_still_running"] = bool(driver.thread and driver.thread.is_alive())
    if record["fault"] or record["worker_still_running"]:
        raise RuntimeError("清理STOP失败或线程未退出；现场确认机构是否停止")
    record["software_execution_complete"] = True
    record["events"].append("PULSE_EXECUTION_COMPLETE")
    save()


def execute_official_once(transport, record, save=lambda: None):
    channel = LoggedSerial(transport, record, announce=False) if transport is not None else None
    start = bytes.fromhex(CONFIG["start_hex"])
    stop = bytes.fromhex(CONFIG["stop_hex"])
    record.update(execution_mode="official_serial_sequence", worker_still_running=False)
    try:
        try:
            record["request_accepted"] = True
            record["pulse_result"] = "pending"
            record["events"].append("OFFICIAL_SEQUENCE_ACCEPTED")
            save()
            try:
                if channel is not None and channel.write(start) != len(start):
                    raise OSError("官方时序START串口短写；不重试")
                time.sleep(CONFIG["pulse_seconds"])
            finally:
                if channel is not None and channel.write(stop) != len(stop):
                    raise OSError("官方时序STOP串口短写；现场确认停止")
        finally:
            if channel is not None:
                channel.close()
    except BaseException as error:
        record["pulse_result"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        record["fault"] = "{}: {}".format(type(error).__name__, error)
        raise
    record["pulse_result"] = "completed"
    record["software_execution_complete"] = True
    record["events"].append("OFFICIAL_SEQUENCE_SOFTWARE_COMPLETE")
    save()


def new_record(live):
    return dict(created_at=datetime.now(timezone.utc).isoformat(),
                mode="live" if live else "simulation", simulated=not live,
                port=PORT if live else None, baudrate=9600, pulse_seconds=CONFIG["pulse_seconds"],
                serial_writes=[],
                status="started", request_accepted=None, pulse_id=None,
                pulse_result="not_requested", software_execution_complete=False,
                device_acknowledged=False, actual_discharge_confirmed=False,
                hit_confirmed=False, physical_stop_confirmed=False, events=[])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="小车上真实单次测试；必须现场输入FIRE")
    mode.add_argument("--check", action="store_true", help="只查路径、权限、占用和pyserial，不打开串口")
    mode.add_argument("--diagnose", action="store_true", help="只读采集串口身份、占用与最近记录；不打开设备、不发射")
    parser.add_argument("--official", action="store_true",
                        help="独立官方串口时序对照：同步START、等0.10秒、STOP；不走脉冲线程，仍须现场确认")
    args = parser.parse_args(argv)
    if args.official and (args.check or args.diagnose):
        parser.error("--official仅用于独立模拟或配合--live，不能与--check/--diagnose混用")
    if args.diagnose:
        from shoot_diagnostics import run_diagnosis
        return run_diagnosis(PROJECT, PORT, check_live)
    if args.check:
        try:
            check_live()
            print("发射器路径：{} → {}，9600波特，8N1。".format(PORT, Path(PORT).resolve()))
            print("环境检查通过；未打开串口，未确认实物连接、实际出弹或物理停止。")
            return 0
        except Exception as error:
            print("检查失败：{}".format(error), file=sys.stderr)
            return 2

    record = new_record(args.live)
    record["execution_mode"] = "official_serial_sequence" if args.official else "pulse_driver"
    session = PROJECT / "runs" / "发射单项" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    path = session / "result.json"

    def save():
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    previous_handler = None
    exit_code = 0
    try:
        session.mkdir(parents=True, exist_ok=False)
        save()
        print("模式：{}。一次0.10秒软件脉冲，不保证只出一颗，也不保证一定出弹。".format(
            "真实串口" if args.live else "模拟，不连接设备"))
        if args.official:
            print("官方串口时序对照：START→等待0.10秒→STOP；不走脉冲线程、不加前置STOP、不自动补射。")
        confirmed = True
        if args.live:
            check_live()
            if not sys.stdin.isatty():
                raise RuntimeError("真实测试必须在现场交互终端确认，不接受管道或自动确认")
            print("仅用于你说明的玩具软弹装置：车已停稳、朝向无人靶区、戴好护目镜。")
            print("停止其他发射/底盘/比赛程序，准备可靠的物理断电手段；本工具不控制底盘。")
            confirmed = input("确认上述条件，输入 FIRE 执行一次；其他输入取消：").strip() == "FIRE"
        if not confirmed:
            record["status"] = "cancelled_before_open"
            print("已取消，未打开串口。")
        else:
            previous_handler = signal.signal(signal.SIGTERM, handle_termination)
            transport = open_serial() if args.live else None
            if args.official:
                execute_official_once(transport, record, save)
            else:
                execute_once(transport, record, save)
            record["status"] = "software_complete" if args.live else "simulation_complete"
            print("{}执行结束；实际出弹、停止和命中均未自动确认。".format(
                "软件串口写入" if args.live else "模拟"))
    except KeyboardInterrupt:
        exit_code = 130
        record["status"] = "interrupted"
        record["error"] = "用户中断；如已连接设备，必须现场确认停止，不重试"
        print(record["error"], file=sys.stderr)
    except Exception as error:
        exit_code = 2
        record["status"] = "failed"
        record["error"] = "{}: {}".format(type(error).__name__, error)
        print("测试失败：{}；不自动重试，也不能据此断定未出弹或已停止。".format(error), file=sys.stderr)
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
    try:
        save()
        print("记录：{}".format(path))
    except OSError as error:
        print("记录保存失败：{}".format(error), file=sys.stderr)
        exit_code = 2
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
