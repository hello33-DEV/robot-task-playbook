"""Pure control logic. No ROS, serial, or implicit hardware side effects."""
import math
from collections import namedtuple


class MissionFault(RuntimeError):
    pass


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('%s must be a finite number' % name)
    return float(value)


def positive(value, name):
    value = number(value, name)
    if value <= 0:
        raise ValueError('%s must be positive' % name)
    return value


def validate_config(c, live=False):
    if c.get('schema_version') != 1:
        raise ValueError('unsupported schema_version')
    for name in ('loop_hz', 'mission_timeout', 'navigation_timeout', 'service_timeout',
                 'recognition_timeout', 'sensor_timeout', 'lease_timeout', 'velocity_timeout',
                 'settle_time', 'shot_wait', 'speech_timeout', 'localization_timeout', 'amcl_pose_timeout',
                 'max_xy_variance', 'max_yaw_variance', 'arrival_xy_tolerance', 'arrival_yaw_tolerance_deg',
                 'fire_max_sample_age', 'fire_max_linear_speed', 'fire_max_angular_speed'):
        positive(c[name], name)
    if c['loop_hz'] < 10 or c['loop_hz'] * min(c['lease_timeout'], c['velocity_timeout']) < 3:
        raise ValueError('watchdog requires >=3 control ticks per timeout and >=10 Hz')
    for name in ('linear_limit', 'angular_limit'):
        positive(c[name], name)
    if c['linear_limit'] > .5 or c['angular_limit'] > .8:
        raise ValueError('speed limits exceed software safety caps (0.5 m/s, 0.8 rad/s)')
    for name in ('pulse_seconds', 'cooldown_seconds', 'write_timeout'):
        positive(c['serial'][name], 'serial.' + name)
    if c['serial']['pulse_seconds'] > .15:
        raise ValueError('pulse_seconds must be <= 0.15')
    if c['shot_wait'] < c['serial']['pulse_seconds'] + c['serial']['cooldown_seconds']:
        raise ValueError('shot_wait must cover pulse and cooldown')
    for key in ('start_hex', 'stop_hex'):
        frame = bytes.fromhex(c['serial'][key])
        if len(frame) != 8 or frame[0] != 0x55 or frame[-1] != sum(frame[:-1]) & 0xff:
            raise ValueError('serial frames must contain 8 bytes with a valid 0x55 header/checksum')
        if frame[2] != (0x12 if key == 'start_hex' else 0x11):
            raise ValueError('START/STOP command bytes do not match supplied hardware protocol')
    positive(c['serial']['baudrate'], 'serial.baudrate')
    for key in ('image_width','image_height'):
        positive(c[key], key)
    if c['fire_max_sample_age'] > c['sensor_timeout']:
        raise ValueError('fire_max_sample_age must not exceed sensor_timeout')
    if c['vision_backend'] in ('api', 'hybrid'):
        positive(c['api_request_timeout'], 'api_request_timeout')
        positive(c['api_total_timeout'], 'api_total_timeout')
        if c['api_total_timeout'] >= c['recognition_timeout']:
            raise ValueError('recognition_timeout must exceed api_total_timeout')
        if type(c['api_attempts']) is not int or not 1 <= c['api_attempts'] <= 3:
            raise ValueError('api_attempts must be 1..3')
    for name, pose in c['waypoints'].items():
        if len(pose) != 3:
            raise ValueError('waypoint %s must be [x_m,y_m,yaw_deg]' % name)
        for value in pose:
            number(value, name)
    for name in ('recognition', 'ring', 'rotating', 'moving', 'finish'):
        route = c['routes'][name]
        if not route or any(p not in c['waypoints'] for p in route):
            raise ValueError('invalid route: ' + name)
    for name in ('ring', 'rotating', 'moving'):
        a = c['aim'][name]
        pulse = positive(a.get('pulse_seconds', c['serial']['pulse_seconds']), name + '.pulse_seconds')
        if pulse > .15 or c['shot_wait'] < pulse + c['serial']['cooldown_seconds']:
            raise ValueError('invalid per-target pulse duration/shot_wait')
        for key in ('timeout', 'tolerance', 'kp', 'max_angular', 'dwell_seconds', 'max_sample_gap'):
            positive(a[key], name + '.' + key)
        for key in ('setpoint', 'prediction_seconds'):
            number(a[key], key)
        if not 0 <= a['prediction_seconds'] <= .5:
            raise ValueError('prediction_seconds must be 0..0.5')
        if a['samples'] < 2 or int(a['samples']) != a['samples']:
            raise ValueError('aim.samples must be an integer >=2')
        if a['max_angular'] > c['angular_limit']:
            raise ValueError('aim angular limit exceeds global limit')
        if a.get('y_window'):
            lo, hi = a['y_window']
            if number(lo, 'window lower') >= number(hi, 'window upper'):
                raise ValueError('y_window must be ascending')
    ids = c['aim']['ring']['object_ids']
    if not ids or any(isinstance(i, bool) or not isinstance(i, int) or i == 255 or i < 0 for i in ids):
        raise ValueError('ring object_ids must contain real IDs, not 255')
    if c.get('ring_message_type') not in ('objects_stamped', 'point'):
        raise ValueError('ring_message_type must be objects_stamped or point')
    if live and c['ring_message_type'] != 'objects_stamped':
        raise ValueError('Live ring detection requires stamped observations, not legacy Point')
    if c['vision_backend'] not in ('api', 'local', 'hybrid', 'legacy'):
        raise ValueError('vision_backend must be api, local, hybrid or legacy')
    if live:
        needed = ('waypoints', 'camera_axes', 'aim_offsets', 'marker_ids', 'serial_protocol',
                  'velocity_mux', 'localization', 'driver_watchdog')
        missing = [k for k in needed if c['calibration'].get(k) is not True]
        if missing:
            raise ValueError('Live mode blocked; calibration unconfirmed: ' + ', '.join(missing))
    return c


def parse_vision(response, success=True):
    import json
    if not success:
        raise MissionFault('vision service reported failure: ' + str(response))
    try:
        if str(response).lstrip().startswith('{'):
            d = json.loads(response)
            a, b = d['math_answer'], d['fruit_id']
        else:
            fields = str(response).split(',')
            if len(fields) != 2:
                raise ValueError('expected exactly two fields')
            a, b = [int(x.strip()) for x in fields]
        if type(a) is not int or type(b) is not int or a not in range(1, 6) or b not in (6, 7, 8):
            raise ValueError('answer outside allowed ranges')
        return a, b
    except (ValueError, TypeError, KeyError) as e:
        raise MissionFault('invalid vision response: %s (%s)' % (response, e))


Observation = namedtuple('Observation', 'target_id x y z received seq')


class AimController:
    """Only distinct fresh samples count. Missing/out-of-window data resets the gate."""
    def __init__(self, config, sensor_timeout):
        self.c = config
        self.sensor_timeout = sensor_timeout
        self.reset()

    def reset(self):
        self.first = None
        self.count = 0
        self.last_seq = None
        self.last_obs = None

    def step(self, obs, now):
        if obs is None or not all(math.isfinite(v) for v in (obs.x, obs.y, obs.z, obs.received)) \
                or not 0 <= now - obs.received <= self.sensor_timeout:
            self.reset()
            return 0.0, False
        if self.last_obs is not None and obs.received - self.last_obs.received > self.c['max_sample_gap']:
            self.reset()
        if self.last_seq == obs.seq:
            # Do not turn one cached frame into a multi-frame trigger.
            return self.last_velocity, False
        if self.last_obs is not None and (obs.target_id != self.last_obs.target_id or obs.received < self.last_obs.received):
            self.reset()
        error = obs.x - self.c['setpoint']
        velocity = 0.0
        if self.last_obs is not None:
            dt = obs.received - self.last_obs.received
            if dt > .001:
                velocity = (obs.x - self.last_obs.x) / dt
        predicted = error + velocity * self.c['prediction_seconds']
        angular = -self.c['kp'] * predicted
        angular = max(-self.c['max_angular'], min(self.c['max_angular'], angular))
        window = self.c.get('y_window')
        y = abs(obs.y) if self.c.get('absolute_y', False) else obs.y
        in_window = not window or window[0] <= y <= window[1]
        aligned = abs(error) <= self.c['tolerance'] + 1e-12 and abs(predicted) <= self.c['tolerance'] + 1e-12 and in_window
        self.last_seq, self.last_obs = obs.seq, obs
        self.last_velocity = 0.0 if abs(error) <= self.c['tolerance'] else angular
        if not aligned:
            self.first, self.count = None, 0
        else:
            if self.first is None:
                self.first = obs.received
            self.count += 1
        ready = aligned and self.count >= self.c['samples'] and obs.received - self.first >= self.c['dwell_seconds']
        return self.last_velocity, ready


class CommandGate:
    """Single velocity owner with wall-clock leases and latched emergency stop."""
    def __init__(self, lease_timeout, velocity_timeout, linear_limit, angular_limit):
        self.lease_timeout, self.velocity_timeout = lease_timeout, velocity_timeout
        self.linear_limit, self.angular_limit = linear_limit, angular_limit
        self.mode, self.lease_at = 'stop', float('-inf')
        self.commands = {}
        self.estop = False

    def lease(self, mode, now):
        if mode not in ('stop', 'navigation', 'manual'):
            raise ValueError('invalid control mode')
        # Expired leases must not revive buffered commands.
        if mode != self.mode or now - self.lease_at > self.lease_timeout:
            self.commands.clear()
        self.mode, self.lease_at = mode, now

    def command(self, source, xyz, now):
        if source not in ('navigation', 'manual') or len(xyz) != 3:
            return
        if not all(math.isfinite(x) for x in xyz):
            self.commands.pop(source, None)
            return
        self.commands[source] = (xyz, now)

    def active(self, now):
        return not self.estop and 0 <= now - self.lease_at <= self.lease_timeout

    def output(self, now):
        if not self.active(now) or self.mode == 'stop':
            return (0.0, 0.0, 0.0)
        xyz, when = self.commands.get(self.mode, ((0, 0, 0), float('-inf')))
        if not 0 <= now - when <= self.velocity_timeout:
            return (0.0, 0.0, 0.0)
        vx, vy, wz = xyz
        norm = math.hypot(vx, vy)
        if norm > self.linear_limit:
            vx, vy = vx * self.linear_limit / norm, vy * self.linear_limit / norm
        return vx, vy, max(-self.angular_limit, min(self.angular_limit, wz))

    def can_fire(self, now):
        cmd = self.commands.get('manual')
        return (self.active(now) and self.mode == 'manual' and cmd is not None
                and 0 <= now - cmd[1] <= self.velocity_timeout
                and all(abs(x) < 1e-6 for x in cmd[0]))

    def emergency_stop(self):
        self.estop = True
        self.commands.clear()
        self.mode = 'stop'

    def reset_estop(self):
        self.estop = False
        self.commands.clear()
        self.mode, self.lease_at = 'stop', float('-inf')


class CompetitionMission:
    """IO methods are injected; the identical sequence runs in simulation and ROS."""
    def __init__(self, config, io):
        self.c, self.io = config, io
        self.deadline = None
        self.shots = []

    def check(self):
        if self.io.aborted():
            raise MissionFault('emergency stop / abort / shutdown')
        if self.deadline is not None and self.io.now() >= self.deadline:
            raise MissionFault('overall mission timeout')

    def wait(self, seconds):
        end = self.io.now() + seconds
        while self.io.now() < end:
            self.check()
            self.io.sleep(min(1.0 / self.c['loop_hz'], end - self.io.now()))

    def route(self, name):
        self.check()
        self.io.state('NAV_' + name.upper())
        for waypoint in self.c['routes'][name]:
            self.check()
            self.io.navigate(waypoint, self.c['waypoints'][waypoint], self.c['navigation_timeout'], self.check)
        self.io.stop()
        self.wait(self.c['settle_time'])
        self.stationary_check()
        self.io.state('ROUTE_COMPLETE', route=name, waypoint=self.c['routes'][name][-1])

    def stationary_check(self):
        self.check()
        if not self.io.stationary():
            raise MissionFault('base moving or odometry unavailable at task point or during speech')

    def announce(self, text):
        self.io.stop()
        self.stationary_check()
        self.io.speak(text, self.stationary_check)
        self.stationary_check()

    def recognize_and_announce(self):
        self.io.stop()
        self.stationary_check()
        self.io.state('RECOGNIZE')
        answer, fruit = self.io.recognize(self.c['recognition_timeout'], self.stationary_check)
        self.stationary_check()
        if type(answer) is not int or answer not in range(1, 6) \
                or type(fruit) is not int or fruit not in (6, 7, 8):
            raise MissionFault('invalid recognition result; answer not announced')
        self.io.state('ANNOUNCE', math_answer=answer, fruit_id=fruit)
        self.announce('识别到旋转靶%d，识别到移动靶%d。' % (answer, fruit))
        self.io.state('RECOGNITION_COMPLETE', math_answer=answer, fruit_id=fruit)
        return answer, fruit

    def aim_and_fire(self, name, target_id):
        self.io.state('AIM_' + name.upper(), target_id=target_id)
        self.io.begin_aim(name, target_id)
        controller = AimController(self.c['aim'][name], self.c['sensor_timeout'])
        deadline = self.io.now() + self.c['aim'][name]['timeout']
        while self.io.now() < deadline:
            self.check()
            obs = self.io.observation(name, target_id)
            allowed_ids = self.c['aim']['ring']['object_ids'] if name == 'ring' else (target_id,)
            if obs is not None and (type(obs.target_id) is not int or obs.target_id not in allowed_ids):
                obs = None
            angular, ready = controller.step(obs, self.io.now())
            self.io.turn(angular)
            if ready and self.io.stationary():
                self.io.turn(0.0)
                # No blocking pause between the final target sample and pulse request.
                self.io.fire(name, obs.target_id)
                self.shots.append(name)
                self.io.state('SHOT_' + name.upper(), target_id=obs.target_id, hit_confirmed=False)
                self.wait(self.c['shot_wait'])
                self.io.stop()
                self.check()
                self.io.state('TARGET_TASK_COMPLETE', task=name, target_id=obs.target_id, hit_confirmed=False)
                return
            self.io.sleep(1.0 / self.c['loop_hz'])
        raise MissionFault('target acquisition/alignment timeout: ' + name)

    def run(self):
        self.deadline = self.io.now() + self.c['mission_timeout']
        try:
            self.io.state('PREFLIGHT')
            self.io.preflight(self.check)
            self.io.stop()
            self.wait(self.c['settle_time'])
            self.io.state('ANNOUNCE_START')
            self.announce('比赛开始。')
            self.route('recognition')
            answer, fruit = self.recognize_and_announce()
            self.route('ring')
            self.aim_and_fire('ring', None)
            self.route('rotating')
            self.aim_and_fire('rotating', answer)
            self.route('moving')
            self.aim_and_fire('moving', fruit)
            self.route('finish')
            self.io.state('ANNOUNCE_FINISH')
            self.announce('到达终点，比赛结束。')
            self.check()
            self.io.state('COMPLETE', shots_commanded=self.shots, hit_confirmed=False)
            return True
        except Exception as e:
            self.io.state('FAULT', error=str(e), shots_commanded=self.shots)
            return False
        finally:
            self.io.cleanup()


def object_centers(data, width=640, height=480):
    """Decode Qt/FindObject homography layout without relying on an unstamped Point."""
    if len(data) % 12:
        raise ValueError('objects array length must be a multiple of 12')
    result, seen = {}, set()
    for offset in range(0, len(data), 12):
        d = data[offset:offset+12]
        if not all(math.isfinite(x) for x in d) or d[0] != int(d[0]) or d[0] < 0 or d[1] <= 0 or d[2] <= 0:
            continue
        key = int(d[0])
        if key in seen:
            result.pop(key, None)  # Ambiguous duplicate identity is not a valid target.
            continue
        seen.add(key)
        points = []
        for x, y in ((0,0), (d[1],0), (d[1],d[2]), (0,d[2])):
            denominator = d[5]*x + d[8]*y + d[11]
            if abs(denominator) < 1e-9:
                break
            px = (d[3]*x+d[6]*y+d[9])/denominator
            py = (d[4]*x+d[7]*y+d[10])/denominator
            if not math.isfinite(px) or not math.isfinite(py) or not 0 <= px < width or not 0 <= py < height:
                break
            points.append((px,py))
        if len(points) != 4:
            continue
        cross = []
        for i in range(4):
            a,b,c = points[i],points[(i+1)%4],points[(i+2)%4]
            cross.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
        if not (all(x > 1 for x in cross) or all(x < -1 for x in cross)):
            continue
        # Match the supplied print_objects_detected_node.cpp calculation.
        # Points are top-left, top-right, bottom-right, bottom-left.
        center_x = (max(points[0][0], points[3][0]) + max(points[1][0], points[2][0])) / 2
        result[key] = (center_x, sum(p[1] for p in points)/4)
    return result
