"""Deterministic fake IO. Never imports ROS, touches serial, or publishes velocity."""
import json
from aic_competition.core import Observation, MissionFault


class SimIO:
    def __init__(self, config, scenario='success', status_callback=None):
        self.c, self.scenario, self.callback = config, scenario, status_callback
        self.t, self.seq, self.target, self.aim_start = 0.0, 0, None, 0.0
        self.states, self.shots, self.visited, self.cleanup_done = [], [], [], False
        self.last_sample_tick = None
        self.cached = None

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds

    def aborted(self):
        return self.scenario == 'estop' and self.t >= 4.0

    def state(self, name, **details):
        item = dict(state=name, elapsed=round(self.t, 3), **details)
        self.states.append(item)
        if self.callback:
            self.callback(json.dumps(item, ensure_ascii=False))

    def preflight(self, check):
        check()

    def navigate(self, name, pose, timeout, check):
        check()
        self.visited.append(name)
        if self.scenario == 'nav_failure' and name == 'ring':
            raise MissionFault('simulated move_base ABORTED')
        for _ in range(10):
            self.sleep(.1)
            check()
        self.state('ARRIVED', waypoint=name)

    def stop(self):
        pass

    def recognize(self, timeout, check):
        check()
        if self.scenario == 'vision_failure':
            raise MissionFault('simulated recognition rejected')
        return 3, 7

    def speak(self, text, check):
        check()
        if self.scenario == 'speech_failure':
            raise MissionFault('simulated TTS failure')
        self.state('SIMULATED_SPEECH', text=text, simulated=True)
        self.sleep(.5)

    def begin_aim(self, name, target_id):
        self.target, self.aim_start = name, self.t
        self.last_sample_tick, self.cached = None, None

    def observation(self, name, target_id):
        if self.scenario == 'target_lost' and name == 'moving':
            return None
        tick = int((self.t-self.aim_start) * 15)
        if tick == self.last_sample_tick:
            return self.cached
        self.last_sample_tick = tick
        self.seq += 1
        cfg = self.c['aim'][name]
        delta = cfg['tolerance']*3 if self.t-self.aim_start < .6 else 0.0
        if self.scenario == 'wrong_alignment' and name == 'ring':
            delta = cfg['tolerance']*3
        y = sum(cfg['y_window'])/2 if cfg.get('y_window') else 0.2
        stamp = self.t-5 if self.scenario == 'stale_sensor' else self.t
        self.cached = Observation(target_id or cfg.get('object_ids', [0])[0], cfg['setpoint']+delta,
                                  y, 1.5, stamp, self.seq)
        return self.cached

    def turn(self, angular):
        if abs(angular) > self.c['angular_limit']:
            raise AssertionError('angular cap exceeded')

    def stationary(self):
        return True

    def fire(self, name, target_id=None):
        if self.cached is None or target_id != self.cached.target_id:
            raise AssertionError('fire target must match the aligned observation')
        if name in self.shots:
            raise AssertionError('duplicate shot')
        pulse_id = 'simulated-' + name
        self.state('PULSE_REQUEST_ACCEPTED', task=name, target_id=target_id, pulse_id=pulse_id,
                   request_accepted=True, software_execution_complete=False, hit_confirmed=False, simulated=True)
        self.state('PULSE_EXECUTION_COMPLETE', task=name, pulse_id=pulse_id,
                   software_execution_complete=True, hit_confirmed=False, simulated=True)
        self.shots.append(name)

    def cleanup(self):
        self.cleanup_done = True
