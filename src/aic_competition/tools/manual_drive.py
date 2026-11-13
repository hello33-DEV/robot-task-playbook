#!/usr/bin/env python3
"""Bounded keyboard control using the supplied ABOT velocity frame; simulation by default."""
import argparse
import math
import os
from pathlib import PurePosixPath
import select
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time


ZERO = (0.0, 0.0, 0.0)
TICK = 0.05
MAX_LINEAR_SPEED = 0.30
MAX_ANGULAR_SPEED = 1.20


def bounded_number(value, maximum, label):
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= maximum:
        raise ValueError("{}必须大于0且不超过{}".format(label, maximum))
    return number


def velocity_frame(velocity):
    if len(velocity) != 3 or any(not math.isfinite(value) for value in velocity):
        raise ValueError("速度必须是三个有限数")
    if (abs(velocity[0]) > MAX_LINEAR_SPEED or abs(velocity[1]) > MAX_LINEAR_SPEED
            or abs(velocity[2]) > MAX_ANGULAR_SPEED):
        raise ValueError("超过手控硬限速")
    payload = b"\xab\xcd\x0d\x11" + struct.pack(
        ">hhhh", *(int(value * 1000) for value in velocity), 0)
    return payload + bytes([sum(payload) & 0xff])


class PulseControl:
    def __init__(self, speed, turn_speed, duration):
        self.speed = bounded_number(speed, MAX_LINEAR_SPEED, "线速度")
        self.turn_speed = bounded_number(turn_speed, MAX_ANGULAR_SPEED, "角速度")
        self.duration = bounded_number(duration, 0.50, "单次动作秒数")
        self.velocity = ZERO
        self.deadline = 0.0
        self.ready = False

    def command(self, key, now):
        if key == " ":
            self.velocity = ZERO
            self.ready = True
            return "零速请求；已允许下一次短动作"
        directions = {"w": (self.speed, 0.0, 0.0),
                      "s": (-self.speed, 0.0, 0.0),
                      "a": (0.0, 0.0, self.turn_speed),
                      "d": (0.0, 0.0, -self.turn_speed)}
        if key in directions and self.ready:
            self.velocity = directions[key]
            self.deadline = now + self.duration
            self.ready = False
            return "短动作请求 {}，软件有效期 {:.2f} 秒；再次运动前按空格".format(key.upper(), self.duration)
        return None

    def current(self, now):
        if now >= self.deadline:
            self.velocity = ZERO
        return self.velocity


def port_users(port):
    if not isinstance(port, str) or not PurePosixPath(port).is_absolute() or "\x00" in port:
        raise ValueError("串口占用检查必须使用绝对设备路径")
    executable = shutil.which("fuser")
    if not executable:
        raise RuntimeError("缺少系统工具fuser，无法检查串口占用；不打开串口")
    result = subprocess.run([executable, port], capture_output=True,
                            text=True, timeout=2)
    if result.returncode not in (0, 1):
        raise RuntimeError("串口占用检查失败：" + result.stderr.strip())
    if result.returncode == 1 and result.stderr.strip():
        raise RuntimeError("串口占用检查报错：" + result.stderr.strip())
    if result.returncode == 1 and result.stdout.strip():
        raise RuntimeError("串口占用检查结果不一致")
    users = {int(value) for value in result.stdout.split()}
    if any(value <= 0 for value in users):
        raise RuntimeError("串口占用检查返回无效进程号")
    if result.returncode == 0 and not users:
        raise RuntimeError("串口占用检查结果不明确")
    return users


def check_port(port, label="底盘"):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("真实串口模式仅用于小车Ubuntu；电脑上请使用默认模拟或--demo")
    if not PurePosixPath(port).is_absolute():
        raise ValueError("串口必须填写绝对路径，不自动猜测ttyUSB编号")
    if not stat.S_ISCHR(os.stat(port).st_mode):
        raise ValueError("指定路径不是字符设备")
    if not os.access(port, os.R_OK | os.W_OK):
        raise PermissionError("当前用户没有串口读写权限；不自动sudo或修改权限")
    users = port_users(port)
    if users:
        raise RuntimeError("串口已被进程占用：{}；先停止其他{}程序".format(sorted(users), label))


class SerialPort:
    def __init__(self, port):
        self.port = port
        self.fd = None

    def __enter__(self):
        import fcntl
        import termios
        check_port(self.port)
        self.fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.ioctl(self.fd, termios.TIOCEXCL)
            others = port_users(self.port) - {os.getpid()}
            if others:
                raise RuntimeError("串口出现并发占用，拒绝发送")
            settings = termios.tcgetattr(self.fd)
            settings[0] = 0
            settings[1] = 0
            settings[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
            settings[3] = 0
            settings[4] = termios.B115200
            settings[5] = termios.B115200
            settings[6][termios.VMIN] = 0
            settings[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, settings)
            self.send(ZERO)
            return self
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def send(self, velocity):
        remaining = velocity_frame(velocity)
        deadline = time.monotonic() + 0.10
        while remaining:
            timeout = deadline - time.monotonic()
            if timeout <= 0 or not select.select([], [self.fd], [], timeout)[1]:
                raise TimeoutError("串口写入超时；物理停车未确认")
            try:
                written = os.write(self.fd, remaining)
            except BlockingIOError:
                continue
            if written <= 0:
                raise OSError("串口写入未完成")
            remaining = remaining[written:]

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.send(ZERO)
        finally:
            os.close(self.fd)
            self.fd = None


class Simulation:
    def __init__(self):
        self.previous = None

    def __enter__(self):
        return self

    def send(self, velocity):
        frame = velocity_frame(velocity)
        if velocity != self.previous:
            print("[模拟，不发串口] 速度={} 帧={}".format(velocity, frame.hex(" ")))
            self.previous = velocity

    def __exit__(self, exc_type, exc_value, traceback):
        self.send(ZERO)


class Keyboard:
    def __enter__(self):
        self.saved = None
        if not sys.stdin.isatty():
            raise RuntimeError("键盘遥控需要交互终端；离线演示请用--demo")
        if os.name != "nt":
            import termios
            import tty
            self.saved = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        return self

    def read(self):
        if os.name == "nt":
            import msvcrt
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in ("\x00", "\xe0"):
                    msvcrt.getwch()
                    return None
                return key.lower()
            time.sleep(TICK)
            return None
        if select.select([sys.stdin], [], [], TICK)[0]:
            return os.read(sys.stdin.fileno(), 1).decode("ascii", errors="ignore").lower()
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        if self.saved is not None:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, self.saved)


def drive_loop(control, transport, keyboard, clock=time.monotonic):
    try:
        transport.send(ZERO)
        while True:
            transport.send(control.current(clock()))
            key = keyboard.read()
            if key in ("q", "\x1b", "\x03"):
                return
            message = control.command(key, clock())
            if message:
                transport.send(ZERO)
                print(message, flush=True)
            transport.send(control.current(clock()))
    finally:
        transport.send(ZERO)


def demo(control):
    with Simulation() as transport:
        transport.send(ZERO)
        for key in "wsad":
            control.command(" ", 0.0)
            control.command(key, 0.0)
            transport.send(control.current(0.0))
            transport.send(control.current(control.duration + TICK))
    print("演示结束：只验证软件和封包，不代表小车已运动或停车")


def handle_termination(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="明确选择真实串口（还需现场输入MOVE）")
    mode.add_argument("--check", action="store_true", help="只检查端口路径/权限/占用，不打开端口")
    mode.add_argument("--demo", action="store_true", help="自动演示四方向，仅模拟，无需键盘")
    parser.add_argument("--port", default="/dev/abotbase")
    parser.add_argument("--speed", type=float, default=0.05, help="m/s，最大0.30")
    parser.add_argument("--turn-speed", type=float, default=0.20, help="rad/s，最大1.20")
    parser.add_argument("--duration", type=float, default=0.25, help="单次秒数，最大0.50")
    args = parser.parse_args(argv)
    try:
        control = PulseControl(args.speed, args.turn_speed, args.duration)
        if args.check:
            check_port(args.port)
            print("路径、权限与占用检查通过；未打开串口，未验证车型、固件或物理急停")
            return 0
        if args.demo:
            demo(control)
            return 0
        if args.live:
            check_port(args.port)
            print("真实运动：确认安全地面、发射器禁用、可靠物理停车手段、无人靠近。")
            print("不得同时运行其他底盘驱动。软件发零速不等于物理停车确认。")
            if input("确认上述条件并输入 MOVE（否则退出）：").strip() != "MOVE":
                print("已取消，未打开串口")
                return 0
        print("模式：{}；空格=零速并允许一次动作，W前/S后/A左转/D右转，Q退出。".format(
            "真实串口" if args.live else "模拟，不连接小车"))
        print("每次短动作自动到期；长按不续期，再动须先按空格。不是松键检测，也不是定距控制。")
        previous_handler = signal.signal(signal.SIGTERM, handle_termination)
        try:
            with Keyboard() as keyboard:
                with (SerialPort(args.port) if args.live else Simulation()) as transport:
                    drive_loop(control, transport, keyboard)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
        print("退出，已请求零速（模拟模式无真实发送）；实际停车仍须现场确认")
        return 0
    except KeyboardInterrupt:
        print("已中断并尝试零速；如曾打开串口，须现场确认物理停车")
        return 130
    except (OSError, RuntimeError, ValueError, EOFError, subprocess.SubprocessError) as error:
        print("遥控失败：{}；不要据此认定车辆已停止".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
