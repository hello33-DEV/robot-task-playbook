"""Execution receipts with fake serial, ROS messages and time; no hardware or API."""
import importlib.util
import json
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_ros_adapter as adapter_tests
from test_core import config, Transport
from test_ros_adapter import NS, Publisher, load_adapter
from aic_competition.core import CommandGate, CompetitionMission, MissionFault, Observation
from aic_competition.pulse import PulseDriver
from aic_competition.simulation import SimIO

ROOT = Path(__file__).resolve().parents[1]


class PulseReceiptTests(unittest.TestCase):
    def driver(self, transport=None):
        settings = config()['serial']
        settings.update(pulse_seconds=.03, cooldown_seconds=.03)
        driver = PulseDriver(settings, transport or Transport())
        self.addCleanup(driver.close)
        return driver

    def done(self, driver):
        driver.thread.join(timeout=1)
        self.assertFalse(driver.thread.is_alive())
        return driver.snapshot()

    def test_completed_has_unique_id_and_stop(self):
        driver = self.driver()
        self.assertEqual(driver.snapshot()['pulse_result'], 'idle')
        driver.arm(True)
        self.assertTrue(driver.request(lambda: True))
        first = self.done(driver)
        self.assertEqual(first['pulse_result'], 'completed')
        self.assertEqual(driver.transport.frames, [driver.start_frame, driver.stop_frame])
        driver.next_allowed = float('-inf')
        self.assertTrue(driver.request(lambda: True))
        second = self.done(driver)
        self.assertNotEqual(first['pulse_id'], second['pulse_id'])

    def test_rejection_does_not_create_receipt(self):
        driver = self.driver()
        self.assertFalse(driver.request(lambda: True))
        self.assertIsNone(driver.snapshot()['pulse_id'])
        driver.arm(True)
        self.assertFalse(driver.request(lambda: False))
        self.assertEqual(driver.snapshot()['pulse_result'], 'idle')

    def test_start_stop_and_short_write_failures_never_complete(self):
        for settings in (dict(fail_start=True), dict(fail_stop=True), dict(short=True)):
            with self.subTest(settings=settings):
                driver = self.driver(Transport(**settings))
                driver.arm(True)
                self.assertTrue(driver.request(lambda: True))
                status = self.done(driver)
                self.assertEqual(status['pulse_result'], 'failed')
                self.assertTrue(status['fault'])
                self.assertFalse(status['armed'])
                self.assertFalse(driver.request(lambda: True))

    def test_cancellation_before_start_is_not_completed(self):
        driver = self.driver()
        driver.arm(True)
        with driver.lock:
            self.assertTrue(driver.request(lambda: True))
            driver.stop()
        self.assertEqual(self.done(driver)['pulse_result'], 'cancelled')
        self.assertNotIn(driver.start_frame, driver.transport.frames)

    def test_interlock_loss_before_start_is_not_completed(self):
        driver = self.driver()
        checks = []
        def interlock():
            checks.append(True)
            return len(checks) == 1
        driver.arm(True)
        self.assertTrue(driver.request(interlock))
        self.assertEqual(self.done(driver)['pulse_result'], 'cancelled')
        self.assertFalse(driver.armed)
        self.assertNotIn(driver.start_frame, driver.transport.frames)

    def test_interlock_loss_during_pulse_disarms_and_stops(self):
        driver = self.driver()
        driver.arm(True)
        self.assertTrue(driver.request(lambda: driver.start_frame not in driver.transport.frames))
        self.assertEqual(self.done(driver)['pulse_result'], 'cancelled')
        self.assertFalse(driver.armed)
        self.assertEqual(driver.transport.frames, [driver.start_frame, driver.stop_frame])

    def test_stop_must_return_before_receipt_completes(self):
        entered, release = threading.Event(), threading.Event()
        transport = Transport()
        original = transport.write
        def write(frame):
            if frame[2] == 0x11:
                entered.set()
                if not release.wait(1):
                    raise IOError('fake STOP held too long')
            return original(frame)
        transport.write = write
        driver = self.driver(transport)
        driver.arm(True)
        self.assertTrue(driver.request(lambda: True))
        try:
            self.assertTrue(entered.wait(1))
            self.assertEqual(driver.pulse_result, 'pending')
            self.assertTrue(driver.busy)
        finally:
            release.set()
        self.assertEqual(self.done(driver)['pulse_result'], 'completed')

    def test_disarm_stop_fault_is_latched(self):
        driver = self.driver(Transport(fail_stop=True))
        driver.arm(True)
        with self.assertRaises(IOError):
            driver.arm(False)
        self.assertEqual(driver.snapshot()['pulse_result'], 'failed')
        with self.assertRaises(RuntimeError):
            driver.arm(True)

    def test_thread_launch_failure_rejects_without_start(self):
        driver = self.driver()
        driver.arm(True)
        with patch('aic_competition.pulse.threading.Thread.start', side_effect=RuntimeError('thread unavailable')):
            self.assertFalse(driver.request(lambda: True))
        self.assertFalse(driver.busy)
        self.assertFalse(driver.armed)
        self.assertEqual(driver.snapshot()['pulse_result'], 'failed')
        self.assertNotIn(driver.start_frame, driver.transport.frames)


class AdapterReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.rospy = load_adapter()

    def setUp(self):
        adapter_tests.AdapterTests.setUp(self)
        adapter_tests.AdapterTests.setup_fire(self)
        self.io.abort_event = threading.Event()
        self.io.monitor = True
        self.clock = [10.]
        self.io.now = lambda: self.clock[0]
        self.stops = []
        self.io.stop = lambda: self.stops.append(True)
        self.accepted_status = dict(self.io.bridge, armed=True, busy=True,
                                    pulse_id='test-pulse', pulse_result='pending')
        self.reply = NS(success=True, message='{"pulse_id":"test-pulse"}')
        self.send_status = True
        self.update = lambda: self.accepted_status.update(busy=False, pulse_result='completed')
        def call(service, *args):
            self.calls.append(service)
            if service.startswith('/competition/fire'):
                if self.send_status:
                    self.io.bridge_cb(NS(data=json.dumps(self.accepted_status)))
                return self.reply
            return NS(success=True, message='armed')
        self.io.call = call
        def sleep(seconds):
            self.clock[0] += seconds
            self.io.markers[7] = Observation(7, .05, .2, 1.5, self.clock[0], 2)
            self.update()
            if self.send_status:
                self.io.bridge_cb(NS(data=json.dumps(self.accepted_status)))
        self.io.sleep = sleep
        def guard():
            if self.io.abort_event.is_set():
                raise MissionFault('abort latched')
        self.io.guard = guard

    def failed(self):
        with self.assertRaises(MissionFault):
            self.io.fire('moving')
        self.assertTrue(self.stops)
        self.assertFalse(any(event['state'] == 'PULSE_EXECUTION_COMPLETE' for event in self.io.events))
        self.assertLessEqual(sum(service.startswith('/competition/fire') for service in self.calls), 1)

    def test_success_waits_for_correlated_completion_not_acceptance(self):
        self.io.fire('moving')
        self.assertGreater(self.clock[0], 10.)
        self.assertEqual([event['state'] for event in self.io.events],
                         ['PULSE_REQUEST_ACCEPTED', 'PULSE_EXECUTION_COMPLETE'])
        self.assertFalse(self.io.events[0]['software_execution_complete'])
        self.assertEqual(self.io.events[1]['pulse_id'], 'test-pulse')
        self.assertTrue(self.io.events[1]['software_execution_complete'])
        self.assertTrue(all(event['hit_confirmed'] is False for event in self.io.events))

    def test_completed_status_before_rpc_response_is_allowed(self):
        self.accepted_status.update(busy=False, pulse_result='completed')
        self.io.fire('moving')
        self.assertEqual(len(self.io.events), 2)

    def test_pre_request_disarmed_snapshot_waits_for_new_status(self):
        self.send_status = False
        def update():
            self.send_status = True
            self.accepted_status.update(busy=False, pulse_result='completed')
        self.update = update
        self.io.fire('moving')
        self.assertEqual(len(self.io.events), 2)

    def test_request_rejection_stops_without_retry(self):
        self.reply = NS(success=False, message='denied')
        self.failed()

    def test_missing_malformed_or_reused_acceptance_id_fails(self):
        for message in ('ok', '{}', '[]', '{"pulse_id":true}', '{"pulse_id":""}', '{"pulse_id":"previous"}'):
            with self.subTest(message=message):
                self.setUp()
                self.io.bridge['pulse_id'] = 'previous'
                self.reply.message = message
                self.failed()

    def test_cancel_fault_disarm_and_estop_fail(self):
        for change in (dict(pulse_result='cancelled', busy=False), dict(pulse_result='failed', fault='START error'),
                       dict(pulse_result='failed', fault='STOP error'), dict(armed=False), dict(estop=True),
                       dict(active=False), dict(mode='stop'), dict(dry_run=True)):
            with self.subTest(change=change):
                self.setUp()
                self.update = lambda: self.accepted_status.update(change)
                self.failed()

    def test_status_loss_fails(self):
        self.update = lambda: setattr(self, 'send_status', False)
        self.failed()

    def test_fresh_pending_receipt_times_out(self):
        self.update = lambda: None
        self.failed()
        self.assertGreater(self.clock[0], 11.)

    def test_old_receipt_cannot_complete_new_request(self):
        self.accepted_status.update(pulse_id=None, pulse_result='idle', busy=False, armed=False)
        self.update = lambda: None
        self.failed()

    def test_replaced_id_restart_and_regression_fail(self):
        for pulse_id in ('unrelated-pulse', None):
            with self.subTest(pulse_id=pulse_id):
                self.setUp()
                self.update = lambda: self.accepted_status.update(pulse_id=pulse_id,
                                                                  pulse_result='idle' if pulse_id is None else 'completed')
                self.failed()

    def test_inconsistent_completed_busy_fails(self):
        self.accepted_status['pulse_result'] = 'completed'
        self.failed()

    def test_target_loss_motion_abort_and_deadline_during_wait_fail(self):
        for failure in ('target', 'motion', 'abort', 'deadline'):
            with self.subTest(failure=failure):
                self.setUp()
                def update():
                    if failure == 'target':
                        self.io.markers.clear()
                    elif failure == 'motion':
                        self.io.stationary = lambda: False
                    elif failure == 'abort':
                        self.io.abort_event.set()
                    else:
                        self.io.guard = lambda: (_ for _ in ()).throw(MissionFault('mission deadline'))
                self.update = update
                self.failed()

    def test_bad_status_invalidates_and_latches_instead_of_reusing_health(self):
        for message in ('broken', 'null', '[]', '{}', '{"dry_run":false}',
                        json.dumps(dict(self.accepted_status, armed='false'))):
            with self.subTest(message=message):
                self.setUp()
                self.io.bridge_cb(NS(data=message))
                self.assertEqual(self.io.bridge, {})
                self.assertEqual(self.io.bridge_at, float('-inf'))
                self.assertTrue(self.io.abort_event.is_set())

    def test_transient_estop_or_serial_fault_stays_latched(self):
        for change in (dict(estop=True), dict(fault='serial failure')):
            with self.subTest(change=change):
                self.setUp()
                self.io.bridge_cb(NS(data=json.dumps(dict(self.accepted_status, **change))))
                self.io.bridge_cb(NS(data=json.dumps(self.accepted_status)))
                self.assertTrue(self.io.abort_event.is_set())

    def test_receipt_failure_blocks_next_task_in_complete_mission(self):
        for task in ('ring', 'rotating', 'moving'):
            with self.subTest(task=task):
                self.setUp()
                io = SimIO(config())
                original = io.fire
                def fire(name, target_id):
                    if name == task:
                        self.update = lambda: self.accepted_status.update(pulse_result='failed', fault='STOP error')
                        self.io.fire('moving')
                    original(name, target_id)
                io.fire = fire
                self.assertFalse(CompetitionMission(io.c, io).run())
                self.assertNotIn(task, io.shots)
                self.assertFalse(any(state['state'] == 'TARGET_TASK_COMPLETE' and state['task'] == task for state in io.states))
                forbidden = {'ring': 'NAV_ROTATING', 'rotating': 'NAV_MOVING', 'moving': 'NAV_FINISH'}[task]
                self.assertFalse(any(state['state'] == forbidden for state in io.states))
                self.assertTrue(io.cleanup_done)


class BridgeReceiptTests(unittest.TestCase):
    def setUp(self):
        adapter, rospy = load_adapter()
        modules = {}
        for name, attrs in {
            'rospy': rospy.__dict__,
            'geometry_msgs.msg': dict(Twist=adapter.Twist),
            'std_msgs.msg': dict(String=lambda value: value, Bool=object),
            'std_srvs.srv': dict(Trigger=object, SetBool=object,
                                TriggerResponse=lambda success, message: NS(success=success, message=message),
                                SetBoolResponse=lambda success, message: NS(success=success, message=message)),
        }.items():
            modules[name] = types.ModuleType(name)
            modules[name].__dict__.update(attrs)
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location('tested_safety_bridge', ROOT/'scripts/safety_bridge.py')
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        self.bridge = self.module.SafetyBridge.__new__(self.module.SafetyBridge)
        self.bridge.c = config()
        self.bridge.gun = PulseDriver(self.bridge.c['serial'], Transport())
        self.addCleanup(self.bridge.gun.close)
        self.bridge.interlock = lambda: True
        self.bridge.gun.arm(True)

    def test_bridge_acceptance_matches_worker_id_and_is_not_completion(self):
        with self.bridge.gun.lock:
            response = self.bridge.fire(None, 'moving')
            self.assertTrue(response.success)
            self.assertEqual(json.loads(response.message)['pulse_id'], self.bridge.gun.pulse_id)
            self.assertEqual(self.bridge.gun.pulse_result, 'pending')
        self.bridge.gun.thread.join(1)
        self.assertEqual(self.bridge.gun.snapshot()['pulse_result'], 'completed')

    def test_bridge_rejection_has_no_receipt(self):
        self.bridge.interlock = lambda: False
        response = self.bridge.fire(None, 'ring')
        self.assertFalse(response.success)
        self.assertIsNone(self.bridge.gun.pulse_id)

    def test_watchdog_publishes_atomic_execution_status_and_current_fault_estop(self):
        self.bridge.lock = threading.RLock()
        self.bridge.closed = threading.Event()
        self.bridge.gate = CommandGate(.4, .2, .2, .5)
        self.bridge.pub = Publisher()
        self.bridge.status_pub = Publisher()
        self.bridge.dry = True
        self.bridge.gun.stop()
        self.bridge.gun.failed = 'STOP fault'
        self.bridge.gun.pulse_result = 'failed'
        self.bridge.gun.pulse_id = 'failed-pulse'
        self.bridge.closed.wait = lambda seconds: self.bridge.closed.set()
        self.bridge.watchdog()
        status = json.loads(self.bridge.status_pub.messages[-1])
        self.assertTrue(status['estop'])
        self.assertEqual(status['pulse_id'], 'failed-pulse')
        self.assertEqual(status['pulse_result'], 'failed')
        self.assertFalse(status['busy'])
        self.assertFalse(status['armed'])


if __name__ == '__main__':
    unittest.main()
