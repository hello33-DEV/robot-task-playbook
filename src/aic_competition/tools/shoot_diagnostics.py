#!/usr/bin/env python3
"""Read-only launcher evidence collection; never open a serial device or send commands."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import uuid

from manual_drive import port_users


def command_output(arguments):
    executable = shutil.which(arguments[0])
    if not executable:
        return {"status": "tool_missing", "tool": arguments[0]}
    try:
        result = subprocess.run([executable] + arguments[1:], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=2)
        return {"status": "ok" if result.returncode == 0 else "failed",
                "returncode": result.returncode, "stdout": result.stdout[:16000],
                "stderr": result.stderr[:2000]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "error": str(error)}


def inspect_device(path):
    result = {"path": str(path), "is_symlink": path.is_symlink()}
    try:
        result["resolved"] = str(path.resolve())
        details = path.stat()
        result.update(status="present", character_device=stat.S_ISCHR(details.st_mode),
                      permissions=oct(stat.S_IMODE(details.st_mode)), uid=details.st_uid,
                      gid=details.st_gid, readable=os.access(str(path), os.R_OK),
                      writable=os.access(str(path), os.W_OK))
        if result["character_device"]:
            result["device_number"] = [os.major(details.st_rdev), os.minor(details.st_rdev)]
            result["udev"] = command_output(
                ["udevadm", "info", "--query=property", "--name=" + str(path)])
            try:
                users = sorted(port_users(str(path)))
                result["occupancy"] = {"status": "visible_users" if users else "no_visible_users",
                                       "pids": users}
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                result["occupancy"] = {"status": "unknown", "error": str(error)}
    except FileNotFoundError:
        result["status"] = "broken_symlink" if result["is_symlink"] else "missing"
    except (OSError, RuntimeError) as error:
        result.update(status="unavailable", error=str(error))
    return result


def summarize_record(data):
    if not isinstance(data, dict):
        raise ValueError("记录根节点不是对象")
    fields = ("created_at", "mode", "status", "recognition_status", "return_heading_status",
              "approach_status", "launch_status", "retreat_status", "shoot_and_retreat_requested")
    summary = {key: data[key] for key in fields if key in data}
    launch = data.get("launch_result") if "launch_result" in data else data
    if not isinstance(launch, dict):
        summary["finding"] = "no_launcher_record"
        return summary
    fields = ("request_accepted", "pulse_result", "software_execution_complete", "simulated",
              "device_acknowledged", "actual_discharge_confirmed", "hit_confirmed")
    summary["launcher"] = {key: launch[key] for key in fields if key in launch}
    writes = launch.get("serial_writes", [])
    if not isinstance(writes, list) or any(not isinstance(entry, dict) for entry in writes):
        raise ValueError("serial_writes不是逐帧对象列表")
    summary["serial_writes"] = [
        {key: entry[key] for key in ("command", "frame_hex", "bytes_requested", "bytes_written", "status")
         if key in entry} for entry in writes]
    starts = [position for position, entry in enumerate(writes) if entry.get("command") == "START"]
    if launch.get("simulated") is True or "simulation" in str(data.get("mode", "")):
        summary["finding"] = "simulation_only"
    elif not starts:
        summary["finding"] = "no_start_record"
    elif len(starts) != 1 or any(writes[position].get("status") != "written" or
                                writes[position].get("bytes_written") != 8 for position in starts):
        summary["finding"] = "start_write_uncertain"
    else:
        stops = [entry for entry in writes[starts[0] + 1:] if entry.get("command") == "STOP"]
        if not stops or any(entry.get("status") != "written" or entry.get("bytes_written") != 8
                            for entry in stops):
            summary["finding"] = "stop_write_uncertain"
        elif launch.get("pulse_result") == "completed" and launch.get("software_execution_complete") is True:
            summary["finding"] = "software_pulse_complete_physical_unknown"
        else:
            summary["finding"] = "software_pulse_incomplete"
    return summary


def recent_records(project):
    records = []
    for group in ("固定路线", "发射单项"):
        try:
            candidates = sorted((project / "runs" / group).glob("*/result.json"),
                                key=lambda path: path.stat().st_mtime, reverse=True)[:20]
        except OSError as error:
            records.append({"group": group, "read_status": "unavailable", "error": str(error)})
            continue
        selected = 0
        live_selected = False
        for path in candidates:
            entry = {"path": str(path), "historical_record": True}
            try:
                path.resolve().relative_to(project.resolve())
                if path.stat().st_size > 2 * 1024 * 1024:
                    raise ValueError("记录超过2MiB，未读取")
                summary = summarize_record(json.loads(path.read_text(encoding="utf-8")))
                is_live = str(summary.get("mode", "")).startswith("live")
                if selected >= 2 and (not is_live or live_selected):
                    continue
                entry.update(summary, read_status="read",
                             modified_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat())
                live_selected = live_selected or is_live
            except (OSError, ValueError, TypeError, RecursionError) as error:
                if selected >= 2:
                    continue
                entry.update(read_status="unavailable", error=str(error))
            records.append(entry)
            selected += 1
            if selected >= 2 and live_selected:
                break
    return records


FINDINGS = {
    "no_launcher_record": "没有发射子记录，不能据此判断实物状态。",
    "simulation_only": "这是模拟记录，不是实物发射证据。",
    "no_start_record": "未找到START记录；可能未进入发射阶段或记录不完整，不能证明从未出弹。",
    "start_write_uncertain": "START写入不完整、失败或次数异常；不要自动补射。",
    "stop_write_uncertain": "START之后的STOP记录缺失或异常；不能确认停止，勿重试，现场检查。",
    "software_pulse_complete_physical_unknown": "软件脉冲结束；设备接收、实际出弹和命中仍未确认。",
    "software_pulse_incomplete": "有START/STOP写入，但软件执行完成条件不齐；不要据此判成功。",
}


def run_diagnosis(project, port, check_live):
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "platform": sys.platform,
              "mode": "read_only_diagnosis", "serial_opened": False, "commands_sent": False,
              "device_identity_confirmed": False, "devices": [],
              "limitations": ["设备列表不会自动认定哪个是发射器，不创建别名或修改权限。",
                              "fuser只反映当前用户可见的占用，空结果不保证全系统无人占用。",
                              "历史记录不代表本次执行；缺记录不证明未执行。",
                              "本报告没有进行发射、ACK、出弹、停止或命中验证。"]}
    try:
        check_live()
        report["preflight"] = {"status": "passed_not_physical_validation"}
    except Exception as error:
        report["preflight"] = {"status": "failed", "error": str(error)}
    if sys.platform.startswith("linux"):
        paths = {Path(port), Path("/dev/abotbase")}
        for pattern in ("ttyUSB*", "ttyACM*", "serial/by-id/*", "serial/by-path/*"):
            paths.update(Path("/dev").glob(pattern))
        ordered = sorted(paths, key=lambda path: (path != Path(port), str(path)))
        report["device_list_truncated"] = len(ordered) > 32
        report["devices"] = [inspect_device(path) for path in ordered[:32]]
        report["usb_listing"] = command_output(["lsusb"])
    else:
        report["platform_note"] = "不是小车Linux环境；未检查车端设备，只有本机历史日志。"
    report["recent_records"] = recent_records(project)
    destination = project / "runs" / "发射诊断" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]) / "diagnosis.json"
    try:
        destination.parent.mkdir(parents=True, exist_ok=False)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as error:
        print("诊断报告保存失败：{}".format(error), file=sys.stderr)
        return 2
    print("只读诊断已采集；未打开串口、不移动、不发射、不调用API。")
    print("环境预检：{}。".format(report["preflight"]))
    if "platform_note" in report:
        print(report["platform_note"])
    for device in report["devices"]:
        print("设备：{} → {}，{}；占用：{}".format(
            device["path"], device.get("resolved", "未知"), device["status"],
            device.get("occupancy", {}).get("status", "未检查")))
    for record in report["recent_records"]:
        print("历史记录：{} [{}] {}".format(record.get("path", record.get("group")),
              record.get("mode", "未知模式"), FINDINGS.get(record.get("finding"), "记录不可用。")))
    print("报告：{}".format(destination))
    print("报告采集完成不是发射验收通过；不自动改端口、不补射。")
    return 0 if sys.platform.startswith("linux") else 2
