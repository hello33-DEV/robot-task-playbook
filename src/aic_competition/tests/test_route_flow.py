"""Task ordering and arrival gates; no robot navigation or physical hit checks."""
import math
import threading
import unittest

from test_core import config
from test_ros_adapter import load_adapter, NS, Publisher, Stamp
from aic_competition.core import CompetitionMission, MissionFault
from aic_competition.simulation import SimIO


class RouteFlowTests(unittest.TestCase):
    def test_all_routes_arrive_before_task_and_next_route(self):
        io = SimIO(config())
        self.assertTrue(CompetitionMission(io.c, io).run())
        expected = [waypoint for route in ('recognition', 'ring', 'rotating', 'moving', 'finish')
                    for waypoint in io.c['routes'][route]]
        self.assertEqual(io.visited, expected)
        self.assertEqual([state['waypoint'] for state in io.states if state['state'] == 'ARRIVED'], expected)
        states = [state['state'] for state in io.states]
        for route in ('recognition', 'ring', 'rotating', 'moving', 'finish'):
            start = states.index('NAV_' + route.upper())
            complete = next(index for index, state in enumerate(io.states)
                            if state['state'] == 'ROUTE_COMPLETE' and state['route'] == route)
            self.assertEqual([state['waypoint'] for state in io.states[start:complete]
                              if state['state'] == 'ARRIVED'], io.c['routes'][route])
            task = {'recognition': 'RECOGNIZE', 'finish': 'ANNOUNCE_FINISH'}.get(route, 'AIM_' + route.upper())
            self.assertLess(complete, states.index(task))
        for task, target, next_route in (('ring', 113, 'rotating'), ('rotating', 3, 'moving'), ('moving', 7, 'finish')):
            complete = next(index for index, state in enumerate(io.states)
                            if state['state'] == 'TARGET_TASK_COMPLETE' and state['task'] == task)
            self.assertEqual(io.states[complete]['target_id'], target)
            self.assertFalse(io.states[complete]['hit_confirmed'])
            self.assertLess(complete, states.index('NAV_' + next_route.upper()))

    def test_failure_at_every_waypoint_blocks_later_tasks(self):
        expected = [waypoint for route in ('recognition', 'ring', 'rotating', 'moving', 'finish')
                    for waypoint in config()['routes'][route]]
        for failed_index in range(len(expected)):
            with self.subTest(index=failed_index, waypoint=expected[failed_index]):
                io = SimIO(config())
                original = io.navigate
                def navigate(name, pose, timeout, check):
                    if len(io.visited) == failed_index:
                        io.visited.append(name)
                        raise MissionFault('injected waypoint failure')
                    original(name, pose, timeout, check)
                io.navigate = navigate
                self.assertFalse(CompetitionMission(io.c, io).run())
                self.assertEqual(io.visited, expected[:failed_index+1])
                self.assertEqual(io.states[-1]['state'], 'FAULT')
                self.assertTrue(io.cleanup_done)
                if expected[failed_index] == 'recognition':
                    self.assertFalse(any(state['state'] == 'RECOGNIZE' for state in io.states))
                self.assertFalse(any(state.get('text') == '到达终点，比赛结束。' for state in io.states))

    def test_moving_after_arrival_blocks_each_task(self):
        for route in ('recognition', 'ring', 'rotating', 'moving', 'finish'):
            with self.subTest(route=route):
                io = SimIO(config())
                last = io.c['routes'][route][-1]
                io.stationary = lambda: not (io.visited and io.visited[-1] == last)
                self.assertFalse(CompetitionMission(io.c, io).run())
                self.assertFalse(any(state['state'] == 'ROUTE_COMPLETE' and state['route'] == route
                                     for state in io.states))
                self.assertEqual(io.states[-1]['state'], 'FAULT')
                self.assertTrue(io.cleanup_done)

    def test_fire_failure_blocks_next_route(self):
        for task, previous, forbidden in (('ring', [], 'NAV_ROTATING'),
                                           ('rotating', ['ring'], 'NAV_MOVING'),
                                           ('moving', ['ring', 'rotating'], 'NAV_FINISH')):
            with self.subTest(task=task):
                io = SimIO(config())
                original = io.fire
                def fire(name, target_id):
                    if name == task:
                        raise MissionFault('injected fire request failure')
                    original(name, target_id)
                io.fire = fire
                self.assertFalse(CompetitionMission(io.c, io).run())
                self.assertEqual(io.shots, previous)
                self.assertNotIn(forbidden, [state['state'] for state in io.states])
                self.assertTrue(io.cleanup_done)


class NavigationArrivalTests(unittest.TestCase):
    def setUp(self):
        self.module, self.ros = load_adapter()
        self.module.GoalStatus = NS(SUCCEEDED=3, ABORTED=4, REJECTED=5, PREEMPTED=2, RECALLED=8, LOST=9)
        self.module.MoveBaseGoal = lambda: NS(target_pose=NS(header=NS(), pose=NS(position=NS(), orientation=NS())))
        self.module.tf.transformations = NS(euler_from_quaternion=lambda quat: (0., 0., 0.))
        self.io = self.module.RosIO.__new__(self.module.RosIO)
        self.io.c, self.io.lock = config(), threading.RLock()
        self.clock, self.status, self.stamp = 0., 3, Stamp(100.)
        self.xyz, self.quat = (1., 2., 0.), (0., 0., 0., 1.)
        self.cancelled, self.events, self.goals = 0, [], []
        self.io.now = lambda: self.clock
        self.io.sleep = lambda seconds: setattr(self, 'clock', self.clock+seconds)
        self.io.state = lambda state, **details: self.events.append(state)
        self.io.mode_pub, self.io.vel_pub = Publisher(), Publisher()
        self.io.nav = NS(send_goal=self.goals.append, get_state=lambda: self.status,
                         cancel_all_goals=self.cancel)
        self.io.tf = NS(getLatestCommonTime=lambda *args: self.stamp,
                        lookupTransform=lambda *args: (self.xyz, self.quat))

    def cancel(self):
        self.cancelled += 1

    def navigate(self, check=lambda: None):
        self.io.navigate('test', (1., 2., 0.), .2, check)

    def assert_stopped_without_arrival(self):
        self.assertGreater(self.cancelled, 0)
        self.assertEqual(self.io.mode, 'stop')
        self.assertNotIn('ARRIVED', self.events)

    def test_valid_arrival_stops_and_records_waypoint(self):
        self.navigate()
        self.assertEqual(self.events, ['ARRIVED'])
        self.assertEqual(self.io.mode, 'stop')
        self.assertEqual(self.goals[0].target_pose.header.frame_id, 'map')
        self.assertEqual(self.goals[0].target_pose.pose.position.x, 1.)

    def test_nan_position_or_invalid_quaternion_never_arrives(self):
        for xyz, quat in (((math.nan, 2., 0.), (0., 0., 0., 1.)),
                          ((1., 2., 0.), (0., 0., 0., 0.)),
                          ((1., 2., 0.), (0., 0., math.inf, 1.))):
            with self.subTest(xyz=xyz, quat=quat):
                self.xyz, self.quat = xyz, quat
                with self.assertRaisesRegex(MissionFault, 'invalid arrival transform'):
                    self.navigate()
                self.assert_stopped_without_arrival()

    def test_stale_future_or_nan_tf_time_never_arrives(self):
        for stamp in (98., 100.1, math.nan):
            with self.subTest(stamp=stamp):
                self.stamp = Stamp(stamp)
                with self.assertRaisesRegex(MissionFault, 'arrival transform stale/future'):
                    self.navigate()
                self.assert_stopped_without_arrival()

    def test_outside_position_or_yaw_tolerance_never_arrives(self):
        for xyz, yaw in (((1.5, 2., 0.), 0.), ((1., 2., 0.), math.pi)):
            with self.subTest(xyz=xyz, yaw=yaw):
                self.xyz = xyz
                self.module.tf.transformations.euler_from_quaternion = lambda quat: (0., 0., yaw)
                with self.assertRaisesRegex(MissionFault, 'outside configured arrival tolerance'):
                    self.navigate()
                self.assert_stopped_without_arrival()

    def test_terminal_action_failures_cancel_navigation(self):
        for status in (4, 5, 2, 8, 9):
            with self.subTest(status=status):
                self.status = status
                with self.assertRaisesRegex(MissionFault, 'navigation test failed'):
                    self.navigate()
                self.assert_stopped_without_arrival()

    def test_active_goal_timeout_cancels_navigation(self):
        self.status = 1
        with self.assertRaisesRegex(MissionFault, 'navigation timeout'):
            self.navigate()
        self.assert_stopped_without_arrival()

    def test_transform_exception_cancels_navigation(self):
        def unavailable(*args):
            raise RuntimeError('injected TF failure')
        self.io.tf.lookupTransform = unavailable
        with self.assertRaisesRegex(RuntimeError, 'injected TF failure'):
            self.navigate()
        self.assert_stopped_without_arrival()

    def test_abort_after_goal_is_sent_cancels_navigation(self):
        def check():
            if self.goals:
                raise MissionFault('injected emergency stop')
        with self.assertRaisesRegex(MissionFault, 'injected emergency stop'):
            self.navigate(check)
        self.assert_stopped_without_arrival()


if __name__ == '__main__':
    unittest.main()
