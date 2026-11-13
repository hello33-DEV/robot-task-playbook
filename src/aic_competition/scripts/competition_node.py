#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS adapter for the tested, hardware-independent competition sequence."""
import json
import math
import os
import threading
import time
from datetime import datetime, timezone
import rospy
import actionlib
import rosgraph
import tf
from actionlib_msgs.msg import GoalStatus
from ar_track_alvar_msgs.msg import AlvarMarkers
from find_object_2d.msg import ObjectsStamped
from geometry_msgs.msg import Point, Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, TriggerResponse, SetBool
from aic_competition.core import CompetitionMission, MissionFault, Observation, parse_vision, validate_config, object_centers


class RosIO:
    def __init__(self, config, state_pub):
        self.c, self.state_pub = config, state_pub
        self.lock = threading.RLock()
        self.abort_event, self.finished = threading.Event(), threading.Event()
        self.mode, self.angular = 'stop', 0.0
        self.rings, self.markers, self.seq = {}, {}, 0
        self.last_ring_stamp, self.last_ar_stamp = 0, 0
        self.aim_since, self.target_id = float('inf'), None
        self.aim_stamp, self.aim_name = 0, None
        self.bridge, self.bridge_at = {}, float('-inf')
        self.odom, self.odom_at = None, float('-inf')
        self.pose, self.pose_at = None, float('-inf')
        self.monitor = False
        self.last_audit = float('-inf')
        self.mode_pub = rospy.Publisher('/competition/control_mode', String, queue_size=1)
        self.vel_pub = rospy.Publisher('/competition/manual_cmd_vel', Twist, queue_size=1)
        self.estop_pub = rospy.Publisher('/competition/estop', Bool, queue_size=1, latch=True)
        self.nav = actionlib.SimpleActionClient(self.c['topics']['move_base'], MoveBaseAction)
        self.tf = tf.TransformListener()
        if self.c['ring_message_type'] == 'objects_stamped':
            rospy.Subscriber(self.c['topics']['ring'], ObjectsStamped, self.objects_cb, queue_size=1, tcp_nodelay=True)
        else:
            rospy.Subscriber(self.c['topics']['ring'], Point, self.ring_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.c['topics']['ar'], AlvarMarkers, self.ar_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/competition/bridge_status', String, self.bridge_cb, queue_size=1)
        rospy.Subscriber('/competition/estop', Bool, lambda m: self.abort_event.set() if m.data else None, queue_size=1)
        rospy.Subscriber(self.c['topics']['odom'], Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber(self.c['topics']['amcl'], PoseWithCovarianceStamped, self.pose_cb, queue_size=1)
        rospy.Service('/competition/abort', Trigger, self.abort_service)
        self.journal = os.path.expanduser(self.c['journal'])
        os.makedirs(os.path.dirname(self.journal), exist_ok=True)
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()
        rospy.on_shutdown(self.shutdown)

    def now(self):
        return time.monotonic()

    def sleep(self, seconds):
        self.abort_event.wait(max(0, seconds))

    def header_fresh(self, header, limit):
        stamp = header.stamp.to_sec()
        return stamp > 0 and -.1 <= (rospy.Time.now()-header.stamp).to_sec() <= limit

    def objects_cb(self, msg):
        with self.lock:
            if not self.header_fresh(msg.header, self.c['sensor_timeout']):
                self.rings.clear()
                return
            stamp = msg.header.stamp.to_nsec()
            if stamp <= self.last_ring_stamp:
                return
            self.last_ring_stamp = stamp
            self.rings.clear()
            if stamp <= self.aim_stamp:
                return
            received = self.now() - (rospy.Time.now()-msg.header.stamp).to_sec()
            try:
                centers = object_centers(msg.objects.data, self.c['image_width'], self.c['image_height'])
            except ValueError:
                return
            for key in self.c['aim']['ring']['object_ids']:
                if key in centers:
                    self.seq += 1
                    x,y = centers[key]
                    self.rings[key] = Observation(key, x, y, key, received, self.seq)

    def ring_cb(self, msg):
        with self.lock:
            self.rings.clear()
            self.seq += 1
            if not all(math.isfinite(x) for x in (msg.x, msg.y, msg.z)):
                self.rings.clear()
                return
            if msg.z == 255:
                self.rings.clear()
            elif msg.z == int(msg.z) and int(msg.z) in self.c['aim']['ring']['object_ids']:
                self.rings[int(msg.z)] = Observation(int(msg.z), msg.x, msg.y, msg.z, self.now(), self.seq)

    def ar_cb(self, msg):
        with self.lock:
            if not self.header_fresh(msg.header, self.c['sensor_timeout']):
                self.markers.clear()
                return
            stamp = msg.header.stamp.to_nsec()
            if stamp <= self.last_ar_stamp:
                return
            self.last_ar_stamp = stamp
            self.markers.clear()  # Missing target is not an indefinitely valid cached observation.
            if stamp <= self.aim_stamp:
                return
            received = self.now() - (rospy.Time.now()-msg.header.stamp).to_sec()
            seen = set()
            for marker in msg.markers:
                if marker.id in seen:
                    self.markers.pop(marker.id, None)
                    continue
                seen.add(marker.id)
                frame = marker.pose.header.frame_id or msg.header.frame_id
                if frame.lstrip('/') != self.c['camera_frame'].lstrip('/'):
                    continue
                p = marker.pose.pose.position
                if not all(math.isfinite(v) for v in (p.x, p.y, p.z)) or p.z <= 0:
                    continue
                self.seq += 1
                self.markers[marker.id] = Observation(marker.id, p.x, p.y, p.z, received, self.seq)

    def bridge_cb(self, msg):
        try:
            value = json.loads(msg.data)
            valid = isinstance(value, dict) and all(type(value.get(key)) is bool for key in
                    ('dry_run', 'active', 'estop', 'armed', 'busy')) \
                and value.get('mode') in ('stop', 'manual', 'navigation') \
                and (value.get('fault') is None or isinstance(value.get('fault'), str)) \
                and value.get('pulse_result') in ('idle', 'pending', 'completed', 'cancelled', 'failed') \
                and (value.get('pulse_id') is None and value['pulse_result'] == 'idle'
                     or isinstance(value.get('pulse_id'), str) and bool(value['pulse_id']))
        except (ValueError, TypeError):
            valid = False
        with self.lock:
            if not valid:
                self.bridge, self.bridge_at = {}, float('-inf')
                if self.monitor:
                    self.abort_event.set()
                return
            self.bridge, self.bridge_at = value, self.now()
            if self.monitor and (value['estop'] or value['fault'] or value['dry_run']):
                self.abort_event.set()

    def pulse_status(self):
        with self.lock:
            value, received = dict(self.bridge), self.bridge_at
        if not 0 <= self.now()-received <= self.c['lease_timeout'] \
                or value.get('dry_run', True) or value.get('estop', True) or value.get('fault'):
            raise MissionFault('pulse bridge status stale/unsafe')
        return value

    def wait_pulse(self, pulse_id, previous_id, name, check):
        serial = self.c['serial']
        deadline = self.now() + 2*serial['write_timeout'] + self.c['aim'][name]['pulse_seconds'] + 3*self.c['lease_timeout']
        seen = False
        while self.now() < deadline:
            check()
            value = self.pulse_status()
            if value.get('pulse_id') == pulse_id:
                seen = True
                if value.get('active') is not True or value.get('mode') != 'manual' or value.get('armed') is not True:
                    raise MissionFault('pulse interlock lost or gun disarmed')
                result = value.get('pulse_result')
                if result == 'completed' and value.get('busy') is False:
                    self.state('PULSE_EXECUTION_COMPLETE', task=name, pulse_id=pulse_id,
                               software_execution_complete=True, hit_confirmed=False)
                    return
                if result != 'pending' or value.get('busy') is not True:
                    raise MissionFault('pulse execution failed/cancelled/inconsistent: ' + str(result))
            elif seen or value.get('pulse_id') != previous_id:
                raise MissionFault('pulse receipt replaced or bridge restarted')
            self.sleep(1.0/self.c['loop_hz'])
        raise MissionFault('pulse execution receipt timed out; no retry sent')

    def odom_cb(self, msg):
        if self.header_fresh(msg.header, self.c['localization_timeout']):
            with self.lock:
                self.odom, self.odom_at = msg, self.now()

    def pose_cb(self, msg):
        if self.header_fresh(msg.header, self.c['amcl_pose_timeout']) and msg.header.frame_id.lstrip('/') == self.c['map_frame']:
            with self.lock:
                self.pose, self.pose_at = msg, self.now()

    def aborted(self):
        if self.abort_event.is_set() or rospy.is_shutdown():
            return True
        if self.monitor:
            with self.lock:
                if self.now()-self.bridge_at > self.c['lease_timeout']*3 \
                        or self.bridge.get('estop') or self.bridge.get('fault') or self.bridge.get('dry_run', True):
                    self.abort_event.set()
            if self.now()-self.last_audit > 1.0:
                self.audit_publishers()
            self.check_localization()
        return self.abort_event.is_set()

    def heartbeat(self):
        period = 1.0/self.c['loop_hz']
        while not self.finished.is_set() and not rospy.is_shutdown():
            with self.lock:
                mode = 'stop' if self.abort_event.is_set() else self.mode
                angular = self.angular if mode == 'manual' else 0.0
            self.mode_pub.publish(String(mode))
            msg = Twist()
            msg.angular.z = angular
            self.vel_pub.publish(msg)
            self.finished.wait(period)

    def state(self, name, **details):
        record = dict(state=name, time=datetime.now(timezone.utc).isoformat(), **details)
        text = json.dumps(record, ensure_ascii=False)
        rospy.loginfo(text)
        self.state_pub.publish(String(text))
        with open(self.journal, 'a', encoding='utf-8') as f:
            f.write(text + '\n')

    def call(self, service, service_type, timeout, check, *args):
        end = self.now() + timeout
        while self.now() < end:
            check()
            try:
                rospy.wait_for_service(service, timeout=min(.15, max(.01, end-self.now())))
                break
            except rospy.ROSException:
                pass
        else:
            raise MissionFault('service unavailable: ' + service)
        check()
        if self.now() >= end:
            raise MissionFault('service discovery timed out; no request sent: ' + service)
        done = threading.Event()
        result = {}
        proxy = rospy.ServiceProxy(service, service_type)
        def invoke():
            try:
                result['value'] = proxy(*args)
            except Exception as e:
                result['error'] = e
            finally:
                done.set()
        threading.Thread(target=invoke, daemon=True).start()
        while not done.wait(.02):
            check()
            if self.now() >= end:
                # ROS service calls have no cancellable RPC deadline. Do not retry a hung call.
                raise MissionFault('service call timed out; no duplicate request sent: ' + service)
        check()
        if self.now() >= end:
            raise MissionFault('late service response rejected: ' + service)
        if 'error' in result:
            raise MissionFault('service %s failed: %s' % (service, result['error']))
        return result['value']

    def audit_publishers(self):
        publishers, _, _ = rosgraph.Master(rospy.get_name()).getSystemState()
        self.last_audit = self.now()
        owners = dict(publishers).get('/cmd_vel', [])
        if owners != ['/competition_safety_bridge']:
            raise MissionFault('/cmd_vel must have only /competition_safety_bridge; got ' + repr(owners))
        raw_nav = dict(publishers).get('/competition/navigation_cmd_vel', [])
        if raw_nav != ['/move_base']:
            raise MissionFault('navigation mux remap is missing or has unexpected publishers: ' + repr(raw_nav))

    def check_localization(self):
        with self.lock:
            if self.now()-self.pose_at > self.c['amcl_pose_timeout'] or self.now()-self.odom_at > self.c['localization_timeout']:
                raise MissionFault('AMCL or odometry stale/missing')
            covariance = self.pose.pose.covariance
            relevant = [covariance[i] for i in (0, 7, 35)]
            if not all(math.isfinite(x) and x >= 0 for x in relevant) or max(relevant[:2]) > self.c['max_xy_variance'] \
                    or relevant[2] > self.c['max_yaw_variance']:
                raise MissionFault('AMCL covariance too large; re-localize before running')
        try:
            stamp = self.tf.getLatestCommonTime(self.c['map_frame'], self.c['base_frame'])
            age = (rospy.Time.now()-stamp).to_sec()
            if age < -.1 or age > self.c['localization_timeout']:
                raise MissionFault('map -> base TF is stale')
        except tf.Exception as e:
            raise MissionFault('map -> base TF unavailable: ' + str(e))

    def preflight(self, check):
        self.mission_check = check
        validate_config(self.c, live=True)
        end = self.now()+self.c['service_timeout']
        while self.now() < end:
            check()
            with self.lock:
                ready = self.now()-self.bridge_at < .5 and not self.bridge.get('dry_run', True) \
                    and not self.bridge.get('fault') and not self.bridge.get('estop')
            if ready:
                break
            self.sleep(.05)
        else:
            raise MissionFault('live safety bridge unavailable or estop latched')
        self.audit_publishers()
        if not rospy.get_param('/abot_driver_bsp/competition_watchdog_enabled', False):
            raise MissionFault('base driver watchdog patch not running; patch source, rebuild, and test')
        self.check_localization()
        while self.now() < end:
            check()
            if self.nav.wait_for_server(rospy.Duration(.1)):
                self.monitor = True
                return
        raise MissionFault('move_base action server unavailable')

    def navigate(self, name, pose, timeout, check):
        check()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.c['map_frame']
        goal.target_pose.header.stamp = rospy.Time.now()
        x, y, yaw_deg = pose
        goal.target_pose.pose.position.x, goal.target_pose.pose.position.y = x, y
        yaw = math.radians(yaw_deg)
        goal.target_pose.pose.orientation.z, goal.target_pose.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
        with self.lock:
            self.mode, self.angular = 'navigation', 0.0
        try:
            self.nav.send_goal(goal)
            end = self.now()+timeout
            while self.now() < end:
                check()
                status = self.nav.get_state()
                if status == GoalStatus.SUCCEEDED:
                    stamp = self.tf.getLatestCommonTime(self.c['map_frame'], self.c['base_frame'])
                    age = (rospy.Time.now()-stamp).to_sec()
                    if not 0 <= age <= self.c['localization_timeout']:
                        raise MissionFault('arrival transform stale/future: ' + name)
                    xyz, quat = self.tf.lookupTransform(self.c['map_frame'], self.c['base_frame'], stamp)
                    if not all(math.isfinite(value) for value in tuple(xyz)+tuple(quat)) \
                            or abs(sum(value*value for value in quat)-1.0) > .01:
                        raise MissionFault('invalid arrival transform: ' + name)
                    actual_yaw = tf.transformations.euler_from_quaternion(quat)[2]
                    yaw_error = abs(math.atan2(math.sin(actual_yaw-yaw), math.cos(actual_yaw-yaw)))
                    if not math.isfinite(actual_yaw) \
                            or math.hypot(xyz[0]-x, xyz[1]-y) > self.c['arrival_xy_tolerance'] \
                            or yaw_error > math.radians(self.c['arrival_yaw_tolerance_deg']):
                        raise MissionFault('move_base success outside configured arrival tolerance: ' + name)
                    check()
                    if self.now() >= end:
                        raise MissionFault('navigation timeout at ' + name)
                    self.stop()
                    self.state('ARRIVED', waypoint=name)
                    return
                if status in (GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED,
                              GoalStatus.RECALLED, GoalStatus.LOST):
                    raise MissionFault('navigation %s failed (status %s)' % (name, status))
                self.sleep(1.0/self.c['loop_hz'])
            raise MissionFault('navigation timeout at ' + name)
        finally:
            self.stop()

    def stop(self):
        with self.lock:
            self.mode, self.angular = 'stop', 0.0
        self.mode_pub.publish(String('stop'))
        self.vel_pub.publish(Twist())
        self.nav.cancel_all_goals()

    def recognize(self, timeout, check):
        if self.c['vision_backend'] == 'local':
            response = self.call('/competition/recognize', Trigger, timeout, check)
            return parse_vision(response.message, response.success)
        if self.c['vision_backend'] == 'hybrid':
            try:
                response = self.call('/competition/recognize', Trigger, min(3.2, timeout), check)
                return parse_vision(response.message, response.success)
            except MissionFault as e:
                rospy.logwarn('Local vision rejected or unavailable; falling back to Doubao API: %s', e)
        # Patched official API node can explicitly reset its cache before a new run.
        # 'legacy' remains compatible with the unmodified official node.
        if self.c['vision_backend'] in ('api', 'hybrid'):
            reset = self.call('/fruit_detection/reset', Trigger, self.c['service_timeout'], check)
            if not reset.success:
                raise MissionFault('vision reset refused: ' + reset.message)
        # Supplied API node is cache-only. It is unsafe to accept old nonzero cache values.
        baseline = self.call('/fruit_detection', Trigger, self.c['service_timeout'], check)
        if baseline.message.strip() not in ('0,0', '0, 0'):
            raise MissionFault('legacy cache not empty; restart identify.py before recognition')
        rospy.set_param('/fruit_detection/capture_after', rospy.Time.now().to_sec())
        rospy.set_param('/detect', 1)  # A parameter, NOT a topic in the supplied identify.py.
        end = self.now()+timeout
        try:
            while self.now() < end:
                check()
                r = self.call('/fruit_detection', Trigger, min(2., end-self.now()), check)
                if r.success and r.message.strip() not in ('0,0', '0, 0'):
                    return parse_vision(r.message, r.success)
                self.sleep(.2)
        finally:
            rospy.set_param('/detect', 255)
        raise MissionFault('API recognition timeout')

    def speak(self, text, check):
        from TTS_audio.srv import StringService
        response = self.call(self.c['topics']['tts_service'], StringService, self.c['speech_timeout'], check, text)
        result = response.result.strip()
        if not result or '错误' in result or '出错' in result or 'error' in result.lower():
            raise MissionFault('speech service reported failure: ' + result)

    def begin_aim(self, name, target_id):
        self.stop()
        if name not in ('ring', 'rotating', 'moving') \
                or (name == 'ring' and target_id is not None) \
                or (name != 'ring' and (type(target_id) is not int \
                    or target_id not in (range(1, 6) if name == 'rotating' else (6, 7, 8)))):
            raise MissionFault('invalid task target binding')
        with self.lock:
            self.aim_since, self.target_id = self.now(), target_id
            self.aim_stamp, self.aim_name = rospy.Time.now().to_nsec(), name
            self.rings.clear()
            self.markers.clear()
            self.mode, self.angular = 'manual', 0.0

    def observation(self, name, target_id):
        with self.lock:
            if name != self.aim_name or target_id != self.target_id:
                return None
            if name == 'ring':
                found = [self.rings[k] for k in self.c['aim']['ring']['object_ids'] if k in self.rings]
                sample = found[0] if len(found) == 1 else None
            else:
                sample = self.markers.get(target_id)
            return sample if sample is not None and sample.received > self.aim_since \
                and all(math.isfinite(value) for value in (sample.x, sample.y, sample.z, sample.received)) \
                and 0 <= self.now()-sample.received <= self.c['sensor_timeout'] else None

    def turn(self, angular):
        with self.lock:
            self.mode, self.angular = 'manual', angular
        msg = Twist()
        msg.angular.z = angular
        self.vel_pub.publish(msg)

    def stationary(self):
        with self.lock:
            if self.odom is None or self.now()-self.odom_at > self.c['localization_timeout']:
                return False
            twist = self.odom.twist.twist
            values = (twist.linear.x, twist.linear.y, twist.angular.z)
            return all(math.isfinite(v) for v in values) and math.hypot(values[0], values[1]) <= self.c['fire_max_linear_speed'] \
                and abs(values[2]) <= self.c['fire_max_angular_speed']

    def fire(self, name, target_id=None):
        self.turn(0.0)
        expected_id = self.target_id if target_id is None else target_id
        def pulse_check():
            self.guard()
            if name != self.aim_name or type(expected_id) is not int:
                raise MissionFault('fire task/target binding mismatch')
            obs = self.observation(name, self.target_id)
            cfg = self.c['aim'][name]
            if obs is None or obs.target_id != expected_id \
                    or not 0 <= self.now()-obs.received <= self.c['fire_max_sample_age'] \
                    or abs(obs.x-cfg['setpoint']) > cfg['tolerance'] + 1e-12:
                raise MissionFault('target moved/lost/changed during fire request')
            y = abs(obs.y) if cfg.get('absolute_y') else obs.y
            window = cfg.get('y_window')
            if window and not window[0] <= y <= window[1]:
                raise MissionFault('target left vertical firing window during fire request')
            if not self.stationary():
                raise MissionFault('base not stationary during fire request')
        try:
            pulse_check()
            armed = self.call('/competition/arm', SetBool, .8, pulse_check, True)
            if not armed.success:
                raise MissionFault('arm failed: ' + armed.message)
            pulse_check()
            with self.lock:
                previous_id = self.bridge.get('pulse_id')
            response = self.call('/competition/fire_' + name, Trigger, .8, pulse_check)
            if not response.success:
                raise MissionFault('pulse rejected: ' + response.message)
            try:
                receipt = json.loads(response.message)
                pulse_id = receipt.get('pulse_id') if isinstance(receipt, dict) else None
                if not isinstance(pulse_id, str) or not pulse_id or pulse_id == previous_id:
                    raise ValueError('missing/duplicate pulse_id')
            except (ValueError, TypeError) as exc:
                raise MissionFault('invalid pulse acceptance receipt: ' + str(exc))
            self.state('PULSE_REQUEST_ACCEPTED', task=name, target_id=expected_id, pulse_id=pulse_id,
                       request_accepted=True, software_execution_complete=False, hit_confirmed=False)
            self.wait_pulse(pulse_id, previous_id, name, pulse_check)
        except Exception:
            self.stop()
            raise

    def guard(self):
        self.mission_check()

    def abort_service(self, _):
        self.abort_event.set()
        self.stop()
        self.estop_pub.publish(Bool(True))
        return TriggerResponse(True, 'abort latched; physical emergency stop remains required')

    def cleanup(self):
        self.stop()
        try:
            self.call('/competition/arm', SetBool, .5, lambda: None, False)
        except Exception as e:
            rospy.logwarn('Disarm RPC unavailable: %s (bridge watchdog also disarms)', e)
        self.finished.set()
        self.thread.join(timeout=.5)

    def shutdown(self):
        self.abort_event.set()
        self.finished.set()
        try:
            self.stop()
            self.estop_pub.publish(Bool(True))
        except Exception:
            pass


def main():
    rospy.init_node('competition_mission')
    c = rospy.get_param('/competition/config')
    dry = rospy.get_param('~dry_run', True)
    validate_config(c, live=not dry)
    pub = rospy.Publisher('/competition/status', String, queue_size=10, latch=True)
    started = threading.Event()
    start_lock = threading.Lock()
    def start(_):
        with start_lock:
            if started.is_set():
                return TriggerResponse(False, 'one-shot mission already started; restart node for a new run')
            started.set()
            return TriggerResponse(True, 'dry simulation requested' if dry else 'mission requested')
    rospy.Service('/competition/start', Trigger, start)
    pub.publish(String(json.dumps({'state':'WAIT_START', 'dry_run':dry})))
    rospy.loginfo('Ready. Explicitly call /competition/start. dry_run=%s', dry)
    while not started.wait(.1):
        if rospy.is_shutdown():
            return
    if dry:
        from aic_competition.simulation import SimIO
        io = SimIO(c, status_callback=lambda text: pub.publish(String(text)))
    else:
        io = RosIO(c, pub)
    ok = CompetitionMission(c, io).run()
    rospy.loginfo('Mission finished: %s', 'COMPLETE' if ok else 'FAULT')
    rospy.spin()


if __name__ == '__main__':
    main()
