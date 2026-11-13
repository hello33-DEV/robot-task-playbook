#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The sole /cmd_vel publisher and sole /dev/shoot owner for this stack."""
import json
import threading
import time
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse
from aic_competition.core import CommandGate, validate_config
from aic_competition.pulse import PulseDriver


class SafetyBridge:
    def __init__(self):
        self.c = rospy.get_param('/competition/config')
        self.dry = rospy.get_param('~dry_run', True)
        validate_config(self.c, live=not self.dry)
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.gate = CommandGate(self.c['lease_timeout'], self.c['velocity_timeout'],
                                self.c['linear_limit'], self.c['angular_limit'])
        transport = None
        if not self.dry:
            import serial
            sc = self.c['serial']
            transport = serial.Serial(sc['port'], sc['baudrate'], timeout=.05,
                                      write_timeout=sc['write_timeout'], exclusive=True)
        self.gun = PulseDriver(self.c['serial'], transport)
        self.gun.stop()  # Explicit STOP on startup; never send START here.
        if self.gun.failed:
            raise RuntimeError('cannot send startup STOP: ' + self.gun.failed)
        # In dry mode even /cmd_vel goes to a disconnected preview topic.
        self.pub = rospy.Publisher('/competition/dry_cmd_vel' if self.dry else '/cmd_vel', Twist, queue_size=1)
        self.status_pub = rospy.Publisher('/competition/bridge_status', String, queue_size=1)
        rospy.Subscriber('/competition/control_mode', String, self.lease, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/competition/navigation_cmd_vel', Twist, self.nav, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/competition/manual_cmd_vel', Twist, self.manual, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/competition/estop', Bool, self.estop, queue_size=1)
        rospy.Service('/competition/arm', SetBool, self.arm)
        rospy.Service('/competition/fire', Trigger, self.fire)
        for target in ('ring', 'rotating', 'moving'):
            rospy.Service('/competition/fire_' + target, Trigger,
                          lambda req, target=target: self.fire(req, target))
        rospy.Service('/competition/reset_estop', Trigger, self.reset)
        rospy.on_shutdown(self.shutdown)
        self.thread = threading.Thread(target=self.watchdog, daemon=True)
        self.thread.start()

    def lease(self, msg):
        with self.lock:
            try:
                self.gate.lease(msg.data, time.monotonic())
            except ValueError:
                self.gate.emergency_stop()

    def accept(self, name, msg):
        with self.lock:
            self.gate.command(name, (msg.linear.x, msg.linear.y, msg.angular.z), time.monotonic())

    def nav(self, msg):
        self.accept('navigation', msg)

    def manual(self, msg):
        self.accept('manual', msg)

    def estop(self, msg):
        if msg.data:
            with self.lock:
                self.gate.emergency_stop()
            self.gun.stop()
            self.pub.publish(Twist())

    def interlock(self):
        with self.lock:
            return not self.closed.is_set() and self.gate.can_fire(time.monotonic())

    def arm(self, req):
        if req.data and not self.interlock():
            return SetBoolResponse(False, 'arm denied: fresh manual zero + active lease required')
        try:
            self.gun.arm(req.data)
            return SetBoolResponse(True, 'armed (dry)' if self.dry and req.data else str(req.data))
        except Exception as e:
            return SetBoolResponse(False, str(e))

    def fire(self, _, target=None):
        duration = self.c['serial']['pulse_seconds'] if target is None else self.c['aim'][target]['pulse_seconds']
        with self.gun.lock:
            ok = self.gun.request(self.interlock, duration)
            receipt = json.dumps(dict(pulse_id=self.gun.pulse_id)) if ok else 'pulse denied: interlock/cooldown/fault'
        return TriggerResponse(ok, receipt)

    def reset(self, _):
        self.gun.stop()
        with self.lock:
            self.gate.reset_estop()
        return TriggerResponse(True, 'estop reset; lease invalid and gun disarmed; restart mission explicitly')

    def watchdog(self):
        period = 1.0/self.c['loop_hz']
        while not self.closed.is_set() and not rospy.is_shutdown():
            now = time.monotonic()
            with self.lock:
                xyz = self.gate.output(now)
                active = self.gate.active(now)
                estop, mode = self.gate.estop, self.gate.mode
            if (not active or estop or mode != 'manual' or self.gun.failed) and self.gun.armed:
                self.gun.stop()
            if self.gun.failed:
                with self.lock:
                    self.gate.emergency_stop()
                    estop = self.gate.estop
                xyz = (0, 0, 0)
            msg = Twist()
            msg.linear.x, msg.linear.y, msg.angular.z = xyz
            self.pub.publish(msg)
            self.status_pub.publish(String(json.dumps(dict(dry_run=self.dry, active=active, estop=estop,
                                        mode=mode, **self.gun.snapshot()))))
            self.closed.wait(period)

    def shutdown(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self.gun.close()
        try:
            for _ in range(3):
                self.pub.publish(Twist())
        except rospy.ROSException:
            pass


if __name__ == '__main__':
    rospy.init_node('competition_safety_bridge')
    try:
        bridge = SafetyBridge()
        rospy.spin()
    except Exception as exc:
        rospy.logfatal('Safety bridge failed: %s', exc)
        raise
