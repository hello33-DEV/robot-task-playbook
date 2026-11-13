import unittest
from unittest.mock import patch

import test_ros_adapter as adapter_tests
from test_core import config
from test_ros_adapter import NS, Observation, Stamp, load_adapter
from aic_competition.core import CompetitionMission, MissionFault
from aic_competition.simulation import SimIO


class TargetSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.rospy = load_adapter()

    def setUp(self):
        adapter_tests.AdapterTests.setUp(self)
        self.io.aim_stamp = 0
        self.io.aim_name = 'moving'

    header = adapter_tests.AdapterTests.header
    objects = adapter_tests.AdapterTests.objects
    ar = adapter_tests.AdapterTests.ar
    setup_fire = adapter_tests.AdapterTests.setup_fire

    def test_duplicate_ar_id_is_ambiguous_even_if_one_pose_is_invalid(self):
        for invalid in (False, True):
            for reverse in (False, True):
                with self.subTest(invalid=invalid, reverse=reverse):
                    self.io.last_ar_stamp = 0
                    message = self.ar()
                    duplicate = self.ar(x=float('nan') if invalid else .06).markers[0]
                    message.markers.append(duplicate)
                    if reverse:
                        message.markers.reverse()
                    self.io.ar_cb(message)
                    self.assertIsNone(self.io.observation('moving', 7))

    def test_capture_before_aim_is_not_relabelled_as_new(self):
        self.io.stop = lambda: None
        self.io.begin_aim('moving', 7)
        self.io.now = lambda: 10.1
        self.io.ar_cb(self.ar(at=99.99))
        self.assertIsNone(self.io.observation('moving', 7))
        self.io.begin_aim('ring', None)
        self.io.now = lambda: 10.2
        self.io.objects_cb(self.objects(at=99.99))
        self.assertIsNone(self.io.observation('ring', None))

    def test_fresh_capture_after_aim_can_be_used(self):
        self.io.stop = lambda: None
        self.io.begin_aim('moving', 7)
        self.io.now = lambda: 10.1
        with patch.object(self.rospy.Time, 'now', return_value=Stamp(100.1)):
            self.io.ar_cb(self.ar(at=100.01))
        self.assertEqual(self.io.observation('moving', 7).target_id, 7)

    def test_source_delay_counts_toward_final_sample_age(self):
        self.setup_fire()
        self.io.ar_cb(self.ar(at=99.7))
        with self.assertRaises(MissionFault):
            self.io.fire('moving')
        self.assertFalse(any(service.startswith('/competition/fire') for service in self.calls))

    def test_future_source_cannot_be_used(self):
        self.io.ar_cb(self.ar(at=100.05))
        self.assertIsNone(self.io.observation('moving', 7))

    def test_latest_wrong_point_clears_previous_ring(self):
        self.io.aim_name = 'ring'
        self.io.ring_cb(NS(x=320., y=240., z=113.))
        self.io.ring_cb(NS(x=320., y=240., z=999.))
        self.assertIsNone(self.io.observation('ring', None))

    def test_multiple_ring_candidates_are_not_silently_selected(self):
        self.io.aim_name = 'ring'
        self.io.c['aim']['ring']['object_ids'] = [113, 114]
        self.io.rings = {key: Observation(key, 320., 240., key, 10., key) for key in (113, 114)}
        self.assertIsNone(self.io.observation('ring', None))

    def test_observation_cannot_cross_task_or_requested_id(self):
        self.io.ar_cb(self.ar(key=7))
        self.io.markers[3] = Observation(3, .105, .2, 1.5, 10., 2)
        self.assertIsNone(self.io.observation('rotating', 3))
        self.assertIsNone(self.io.observation('moving', 3))

    def test_invalid_or_future_cached_sample_cannot_fire(self):
        for value in (float('nan'), float('inf'), 10.1):
            with self.subTest(received=value):
                self.setup_fire(Observation(7, .05, .2, 1.5, value, 1))
                with self.assertRaises(MissionFault):
                    self.io.fire('moving')
                self.assertFalse(any(service.startswith('/competition/fire') for service in self.calls))

    def test_ring_cannot_switch_identity_after_alignment(self):
        self.setup_fire()
        self.io.aim_name = 'ring'
        self.io.target_id = None
        self.io.c['aim']['ring']['object_ids'] = [113, 114]
        self.io.rings = {114: Observation(114, 320., 240., 114, 10., 2)}
        with self.assertRaises(MissionFault):
            self.io.fire('ring', 113)
        self.assertEqual(self.calls, [])

    def test_all_target_kinds_pass_only_their_bound_request(self):
        for task, ids in (('ring', (113,)), ('rotating', range(1, 6)), ('moving', (6, 7, 8))):
            for target_id in ids:
                with self.subTest(task=task, target_id=target_id):
                    self.setup_fire()
                    self.io.begin_aim(task, None if task == 'ring' else target_id)
                    self.io.now = lambda: 10.1
                    cfg = self.io.c['aim'][task]
                    sample = Observation(target_id, cfg['setpoint'], .2, 1.5, 10.1, 1)
                    samples = self.io.rings if task == 'ring' else self.io.markers
                    samples[target_id] = sample
                    self.io.fire(task, target_id)
                    self.assertEqual(self.calls, ['/competition/arm', '/competition/fire_' + task])
                    self.io.now = lambda: 10.

    def test_invalid_task_target_binding_is_rejected(self):
        self.io.stop = lambda: None
        for task, target_id in (('ring', 113), ('rotating', 7), ('moving', 3),
                                ('moving', None), ('rotating', True), ('unknown', 7)):
            with self.subTest(task=task, target_id=target_id):
                with self.assertRaises(MissionFault):
                    self.io.begin_aim(task, target_id)

    def test_target_loss_while_waiting_for_fire_service_suppresses_request(self):
        self.setup_fire()
        self.io.call = lambda service, *args: NS(success=True, message='ok') if service.endswith('/arm') else self.module.RosIO.call(self.io, service, *args)
        requests = []
        self.rospy.wait_for_service = lambda *args, **kwargs: self.io.markers.clear()
        self.rospy.ServiceProxy = lambda *args: lambda *values: requests.append(1) or NS(success=True)
        with self.assertRaises(MissionFault):
            self.io.fire('moving')
        self.assertEqual(requests, [])

    def test_rejected_fire_requests_stop_instead_of_leaving_manual_mode(self):
        self.setup_fire()
        stopped = []
        self.io.stop = lambda: stopped.append(True)
        self.io.markers.clear()
        with self.assertRaises(MissionFault):
            self.io.fire('moving')
        self.assertTrue(stopped)

    def test_service_timeout_during_discovery_sends_no_rpc(self):
        clock = [10.]
        self.io.now = lambda: clock[0]
        self.rospy.wait_for_service = lambda *args, **kwargs: clock.__setitem__(0, 11.)
        requests = []
        self.rospy.ServiceProxy = lambda *args: lambda *values: requests.append(1)
        with self.assertRaises(MissionFault):
            self.module.RosIO.call(self.io, '/test', object, .8, lambda: None)
        self.assertEqual(requests, [])


class MissionTargetBindingTests(unittest.TestCase):
    def test_all_fifteen_results_reach_correct_fire_target(self):
        for answer in range(1, 6):
            for fruit in (6, 7, 8):
                with self.subTest(answer=answer, fruit=fruit):
                    io = SimIO(config())
                    io.recognize = lambda *args: (answer, fruit)
                    requests = []
                    def fire(name, target_id):
                        requests.append((name, target_id))
                    io.fire = fire
                    self.assertTrue(CompetitionMission(io.c, io).run())
                    self.assertEqual(requests, [('ring', 113), ('rotating', answer), ('moving', fruit)])

    def test_wrong_observation_identity_never_reaches_fire(self):
        for task in ('ring', 'rotating', 'moving'):
            with self.subTest(task=task):
                io = SimIO(config())
                original = io.observation
                def observation(name, target_id):
                    sample = original(name, target_id)
                    return sample._replace(target_id=999) if name == task else sample
                io.observation = observation
                self.assertFalse(CompetitionMission(io.c, io).run())
                self.assertNotIn(task, io.shots)
                self.assertTrue(io.cleanup_done)


if __name__ == '__main__':
    unittest.main()
