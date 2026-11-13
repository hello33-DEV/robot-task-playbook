import copy
import importlib.util
import math
import sys
import threading
import time
import unittest
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aic_competition.core import (AimController, CommandGate, CompetitionMission, MissionFault,
                                  Observation, object_centers, parse_vision, validate_config)
from aic_competition.simulation import SimIO
from aic_competition.pulse import PulseDriver


def config():
    return yaml.safe_load((ROOT/'config/competition.yaml').read_text(encoding='utf-8-sig'))


class ConfigTests(unittest.TestCase):
    def test_default_mission_limit_matches_supplied_rules(self):
        self.assertEqual(config()['mission_timeout'], 120.0)

    def test_example_valid_offline(self):
        validate_config(config())

    def test_hybrid_backend_is_valid(self):
        c=config();c['vision_backend']='hybrid'
        validate_config(c)

    def test_example_refuses_live(self):
        with self.assertRaisesRegex(ValueError,'calibration'):
            validate_config(config(),live=True)

    def test_live_requires_every_calibration(self):
        c=config()
        for k in c['calibration']:
            c['calibration'][k]=True
        validate_config(c,live=True)

    def test_does_not_accept_truthy_string(self):
        c=config()
        c['calibration']={k:'true' for k in c['calibration']}
        with self.assertRaises(ValueError):
            validate_config(c,live=True)

    def test_bad_route(self):
        c=config();c['routes']['finish']=['missing']
        with self.assertRaises(ValueError):validate_config(c)

    def test_nan_waypoint(self):
        c=config();c['waypoints']['ring'][0]=float('nan')
        with self.assertRaises(ValueError):validate_config(c)

    def test_speed_caps(self):
        c=config();c['linear_limit']=10
        with self.assertRaises(ValueError):validate_config(c)

    def test_invalid_pulse(self):
        c=config();c['serial']['pulse_seconds']=.5
        with self.assertRaises(ValueError):validate_config(c)

    def test_invalid_frame(self):
        c=config();c['serial']['stop_hex']='00'
        with self.assertRaises(ValueError):validate_config(c)

    def test_unknown_ring_id(self):
        c=config();c['aim']['ring']['object_ids']=[255]
        with self.assertRaises(ValueError):validate_config(c)

    def test_point_is_not_live_safe(self):
        c=config();c['ring_message_type']='point'
        with self.assertRaisesRegex(ValueError,'stamped'):validate_config(c,live=True)


class VisionResponseTests(unittest.TestCase):
    def test_legacy(self):self.assertEqual(parse_vision('3,7'),(3,7))
    def test_json(self):self.assertEqual(parse_vision('{"math_answer":5,"fruit_id":8}'),(5,8))
    def test_reject_empty_cache(self):
        with self.assertRaises(MissionFault):parse_vision('0,0')
    def test_service_failed(self):
        with self.assertRaises(MissionFault):parse_vision('3,7',False)
    def test_reject_extra_fields(self):
        with self.assertRaises(MissionFault):parse_vision('3,7,6')
    def test_reject_float_or_bool(self):
        for response in ('3.2,7','{"math_answer":true,"fruit_id":8}','{"math_answer":2.5,"fruit_id":6}'):
            with self.subTest(response=response):
                with self.assertRaises(MissionFault):parse_vision(response)


class AimTests(unittest.TestCase):
    def setUp(self):
        self.c=config()['aim']['moving']
        self.a=AimController(self.c,.35)
    def obs(self,seq,at,x=.05,y=.2,key=7):return Observation(key,x,y,1.5,at,seq)
    def test_turn_sign_and_limit(self):
        vel,ready=self.a.step(self.obs(1,0,5),0)
        self.assertEqual(vel,-self.c['max_angular']);self.assertFalse(ready)
    def test_stable_distinct_samples_fire(self):
        for i in range(3):vel,ready=self.a.step(self.obs(i,i*.05),i*.05)
        self.assertTrue(ready);self.assertEqual(vel,0)
    def test_repeated_frame_not_multiple_samples(self):
        obs=self.obs(1,0)
        for t in [0,.05,.1,.15,.2]:self.assertFalse(self.a.step(obs,t)[1])
    def test_stale_frame_stops(self):
        self.assertEqual(self.a.step(self.obs(1,0,2),1),(0.,False))
    def test_missing_stops(self):self.assertEqual(self.a.step(None,0),(0.,False))
    def test_nan_stops(self):self.assertEqual(self.a.step(self.obs(1,0,float('nan')),0),(0.,False))
    def test_future_sample_stops(self):self.assertEqual(self.a.step(self.obs(1,1),0),(0.,False))
    def test_window_required(self):
        self.c['y_window']=[.18,.24]
        for i in range(5):self.assertFalse(self.a.step(self.obs(i,i*.05,y=.8),i*.05)[1])
    def test_gate_resets_after_misalignment(self):
        self.a.step(self.obs(1,0),0);self.a.step(self.obs(2,.05),.05)
        self.a.step(self.obs(3,.1,1),.1)
        self.assertFalse(self.a.step(self.obs(4,.15),.15)[1])
    def test_identity_change_resets(self):
        self.a.step(self.obs(1,0,key=6),0);self.a.step(self.obs(2,.05,key=6),.05)
        self.assertFalse(self.a.step(self.obs(3,.1,key=7),.1)[1])
    def test_dropped_frame_gap_resets(self):
        self.a.step(self.obs(1,0),0);self.a.step(self.obs(2,.05),.05)
        self.assertFalse(self.a.step(self.obs(3,.3),.3)[1])
    def test_signed_not_absolute_alignment(self):
        for i in range(5):self.assertFalse(self.a.step(self.obs(i,i*.05,x=-.05),i*.05)[1])
    def test_optional_lead_rejects_fast_crossing(self):
        self.c['prediction_seconds']=.2
        self.a.step(self.obs(1,0,.04),0)
        self.assertFalse(self.a.step(self.obs(2,.05,.05),.05)[1])


class GateTests(unittest.TestCase):
    def setUp(self):self.g=CommandGate(.4,.25,.25,.5)
    def test_start_zero(self):self.assertEqual(self.g.output(0),(0.,0.,0.))
    def test_route_owner(self):
        self.g.lease('navigation',0);self.g.command('manual',(.1,0,0),0)
        self.assertEqual(self.g.output(0),(0.,0.,0.))
    def test_fresh_navigation(self):
        self.g.lease('navigation',0);self.g.command('navigation',(.1,0,0),0)
        self.assertEqual(self.g.output(.1),(.1,0,0))
    def test_vector_and_angular_limits(self):
        self.g.lease('manual',0);self.g.command('manual',(1,1,2),0)
        x,y,z=self.g.output(.1)
        self.assertAlmostEqual(math.hypot(x,y),.25);self.assertEqual(z,.5)
    def test_velocity_timeout(self):
        self.g.lease('manual',0);self.g.command('manual',(.1,0,0),0)
        self.assertEqual(self.g.output(.26),(0.,0.,0.))
    def test_lease_timeout(self):
        self.g.lease('manual',0);self.g.command('manual',(.1,0,0),.3)
        self.assertEqual(self.g.output(.41),(0.,0.,0.))
    def test_mode_switch_discards_buffer(self):
        self.g.lease('manual',0);self.g.command('navigation',(.1,0,0),0)
        self.g.lease('navigation',.01)
        self.assertEqual(self.g.output(.01),(0.,0.,0.))
    def test_expired_lease_discards_buffer(self):
        self.g.lease('manual',0);self.g.command('manual',(.1,0,0),.35)
        self.g.lease('manual',.41)
        self.assertEqual(self.g.output(.41),(0.,0.,0.))
    def test_estop_latches_across_lease(self):
        self.g.emergency_stop();self.g.lease('manual',0);self.g.command('manual',(.1,0,0),0)
        self.assertEqual(self.g.output(0),(0.,0.,0.));self.assertFalse(self.g.can_fire(0))
    def test_reset_requires_new_lease_and_zero(self):
        self.g.emergency_stop();self.g.reset_estop()
        self.assertFalse(self.g.can_fire(0))
        self.g.lease('manual',0);self.assertFalse(self.g.can_fire(0))
        self.g.command('manual',(0,0,0),0);self.assertTrue(self.g.can_fire(0))
    def test_cannot_fire_while_turning(self):
        self.g.lease('manual',0);self.g.command('manual',(0,0,.1),0)
        self.assertFalse(self.g.can_fire(0))
    def test_cannot_fire_in_navigation(self):
        self.g.lease('navigation',0);self.g.command('manual',(0,0,0),0)
        self.assertFalse(self.g.can_fire(0))
    def test_nonfinite_command_rejected(self):
        self.g.lease('manual',0);self.g.command('manual',(float('nan'),0,0),0)
        self.assertEqual(self.g.output(0),(0.,0.,0.))


class ObjectTests(unittest.TestCase):
    def data(self):return [113,100,100,1,0,0,0,1,0,200,100,1]
    def test_qt_layout(self):self.assertEqual(object_centers(self.data()),{113:(250.,150.)})
    def test_no_objects(self):self.assertEqual(object_centers([]),{})
    def test_malformed(self):
        with self.assertRaises(ValueError):object_centers([1,2])
    def test_invalid_homography(self):
        d=self.data();d[-1]=0;self.assertEqual(object_centers(d),{})
    def test_duplicate_identity_ambiguous(self):self.assertEqual(object_centers(self.data()*2),{})
    def test_offscreen_rejected(self):
        d=self.data();d[9]=900;self.assertEqual(object_centers(d),{})
    def test_nan_rejected(self):
        d=self.data();d[5]=float('nan');self.assertEqual(object_centers(d),{})


class MissionTests(unittest.TestCase):
    def run_scenario(self,scenario,c=None):
        c=c or config();io=SimIO(c,scenario);ok=CompetitionMission(c,io).run()
        self.assertTrue(io.cleanup_done);return ok,io
    def test_complete_sequence_once(self):
        ok,io=self.run_scenario('success');self.assertTrue(ok)
        self.assertEqual(io.shots,['ring','rotating','moving'])
        self.assertEqual(io.states[-1]['state'],'COMPLETE')
        self.assertEqual(io.visited[-1],'finish')
    def test_start_and_finish_announcements(self):
        io=SimIO(config())
        announcements=[]
        original_speak=io.speak
        def record_speech(text,check):
            announcements.append((text,list(io.visited)))
            original_speak(text,check)
        io.speak=record_speech
        self.assertTrue(CompetitionMission(io.c,io).run())
        self.assertEqual(len(announcements),3)
        self.assertEqual(announcements[0],('比赛开始。',[]))
        self.assertEqual(announcements[1][1],['recognition'])
        self.assertEqual(announcements[-1][0],'到达终点，比赛结束。')
        self.assertEqual(announcements[-1][1][-1],'finish')
    def test_finish_speech_failure_is_not_complete(self):
        io=SimIO(config())
        original_speak=io.speak
        def fail_finish(text,check):
            if io.visited and io.visited[-1]=='finish':
                raise MissionFault('simulated final speech failure')
            original_speak(text,check)
        io.speak=fail_finish
        self.assertFalse(CompetitionMission(io.c,io).run())
        self.assertEqual(io.states[-1]['state'],'FAULT')
        self.assertTrue(io.cleanup_done)
    def test_finish_speech_cannot_overrun_mission_deadline(self):
        io=SimIO(config())
        original_speak=io.speak
        def slow_finish(text,check):
            original_speak(text,check)
            if io.visited and io.visited[-1]=='finish':
                io.sleep(io.c['mission_timeout'])
        io.speak=slow_finish
        self.assertFalse(CompetitionMission(io.c,io).run())
        self.assertEqual(io.states[-1]['state'],'FAULT')
        self.assertTrue(io.cleanup_done)
    def test_navigation_failure_no_shot(self):
        ok,io=self.run_scenario('nav_failure');self.assertFalse(ok);self.assertEqual(io.shots,[])
    def test_recognition_failure_no_shot(self):
        ok,io=self.run_scenario('vision_failure');self.assertFalse(ok);self.assertEqual(io.shots,[])
    def test_speech_failure_no_shot(self):
        ok,io=self.run_scenario('speech_failure');self.assertFalse(ok);self.assertEqual(io.shots,[])
    def test_target_lost_no_blind_fire(self):
        ok,io=self.run_scenario('target_lost');self.assertFalse(ok);self.assertEqual(io.shots,['ring','rotating'])
    def test_bad_alignment_no_fire(self):
        ok,io=self.run_scenario('wrong_alignment');self.assertFalse(ok);self.assertEqual(io.shots,[])
    def test_stale_sensor_no_fire(self):
        ok,io=self.run_scenario('stale_sensor');self.assertFalse(ok);self.assertEqual(io.shots,[])
    def test_estop_no_fire(self):
        ok,io=self.run_scenario('estop');self.assertFalse(ok);self.assertEqual(io.shots,[])
    def test_overall_deadline(self):
        c=config();c['mission_timeout']=3
        ok,io=self.run_scenario('success',c);self.assertFalse(ok);self.assertEqual(io.shots,[])
    def test_moving_base_cannot_fire(self):
        c=config();io=SimIO(c);io.stationary=lambda:False
        self.assertFalse(CompetitionMission(c,io).run());self.assertEqual(io.shots,[])


class Transport:
    def __init__(self,fail_start=False,fail_stop=False,short=False):
        self.frames=[];self.closed=False
        self.fail_start,self.fail_stop,self.short=fail_start,fail_stop,short
    def write(self,frame):
        self.frames.append(frame)
        if (frame[2]==0x12 and self.fail_start) or (frame[2]==0x11 and self.fail_stop):raise IOError('injected serial failure')
        return len(frame)-1 if self.short else len(frame)
    def close(self):self.closed=True


class PulseTests(unittest.TestCase):
    def driver(self,t=None):
        c=config()['serial'];c['pulse_seconds']=.03;c['cooldown_seconds']=.03
        return PulseDriver(c,t or Transport())
    def done(self,d):
        if d.thread:d.thread.join(timeout=1)
        self.assertFalse(d.busy)
    def test_disarmed_no_start(self):
        d=self.driver();self.assertFalse(d.request(lambda:True));self.assertEqual(d.transport.frames,[])
    def test_success_ends_with_stop(self):
        d=self.driver();d.arm(True);self.assertTrue(d.request(lambda:True));self.done(d)
        self.assertEqual([x[2] for x in d.transport.frames],[0x12,0x11]);d.close()
    def test_busy_or_cooldown_no_repeat(self):
        d=self.driver();d.arm(True);self.assertTrue(d.request(lambda:True))
        self.assertFalse(d.request(lambda:True));self.done(d);self.assertFalse(d.request(lambda:True));d.close()
    def test_interlock_rejects(self):
        d=self.driver();d.arm(True);self.assertFalse(d.request(lambda:False));d.close()
        self.assertNotIn(d.start_frame,d.transport.frames)
    def test_start_error_still_attempts_stop(self):
        d=self.driver(Transport(fail_start=True));d.arm(True);d.request(lambda:True);self.done(d)
        self.assertEqual(d.transport.frames[-1],d.stop_frame);self.assertFalse(d.armed);self.assertIsNotNone(d.failed);d.close()
    def test_stop_error_latches(self):
        d=self.driver(Transport(fail_stop=True));d.arm(True);d.request(lambda:True);self.done(d)
        self.assertIsNotNone(d.failed)
        with self.assertRaises(RuntimeError):d.arm(True)
        d.close()
    def test_short_write_latches(self):
        d=self.driver(Transport(short=True));d.arm(True);d.request(lambda:True);self.done(d)
        self.assertIsNotNone(d.failed);d.close()
    def test_abort_sends_stop_and_disarms(self):
        d=self.driver();d.arm(True);d.request(lambda:True);d.stop();self.done(d)
        self.assertFalse(d.armed);self.assertEqual(d.transport.frames[-1],d.stop_frame);d.close()
    def test_close_releases_port(self):
        d=self.driver();d.close();self.assertTrue(d.transport.closed)

    def test_invalid_override_never_starts(self):
        d=self.driver();d.arm(True)
        for duration in (0,-.1,.16,float('nan')):
            self.assertFalse(d.request(lambda:True,duration))
        self.assertNotIn(d.start_frame,d.transport.frames);d.close()

    def test_per_target_duration_reaches_worker(self):
        d=self.driver();received=[]
        d._pulse=lambda interlock,duration:received.append(duration)
        d.arm(True);self.assertTrue(d.request(lambda:True,.1))
        d.thread.join(timeout=1)
        self.assertEqual(received,[.1]);d.close()


if __name__=='__main__':unittest.main()
