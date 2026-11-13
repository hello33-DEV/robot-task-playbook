"""ROS adapter tests with small message/RPC doubles, not a substitute for roscore tests."""
import importlib.util
import json
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aic_competition.core import MissionFault, Observation
NS=types.SimpleNamespace


class Stamp:
    def __init__(self,value):self.value=value
    def to_sec(self):return self.value
    def to_nsec(self):return int(self.value*1e9)
    def __sub__(self,other):return Stamp(self.value-other.value)
    @staticmethod
    def now():return Stamp(100.)


class Twist:
    def __init__(self):self.linear=NS(x=0.,y=0.,z=0.);self.angular=NS(x=0.,y=0.,z=0.)


class Publisher:
    def __init__(self):self.messages=[]
    def publish(self,msg):self.messages.append(msg)


def load_adapter():
    modules={}
    def mod(name,**attrs):
        m=types.ModuleType(name);m.__dict__.update(attrs);modules[name]=m;return m
    rospy=mod('rospy',Time=Stamp,Duration=lambda x:x,is_shutdown=lambda:False,
              wait_for_service=lambda *a,**k:None,ROSException=RuntimeError)
    mod('actionlib');mod('rosgraph');mod('tf',Exception=RuntimeError)
    for package,attrs in {
        'actionlib_msgs.msg':{'GoalStatus':NS()},
        'ar_track_alvar_msgs.msg':{'AlvarMarkers':object},
        'find_object_2d.msg':{'ObjectsStamped':object},
        'geometry_msgs.msg':{'Point':object,'Twist':Twist,'PoseWithCovarianceStamped':object},
        'nav_msgs.msg':{'Odometry':object},
        'move_base_msgs.msg':{'MoveBaseAction':object,'MoveBaseGoal':object},
        'std_msgs.msg':{'String':lambda x:x,'Bool':lambda x:x},
        'std_srvs.srv':{'Trigger':object,'TriggerResponse':object,'SetBool':object},
    }.items():
        mod(package.split('.')[0]);mod(package,**attrs)
    with patch.dict(sys.modules,modules):
        spec=importlib.util.spec_from_file_location('tested_competition_adapter',ROOT/'scripts/competition_node.py')
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    return adapter,rospy


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.module,cls.rospy=load_adapter()
    def setUp(self):
        self.io=self.module.RosIO.__new__(self.module.RosIO)
        self.io.c=yaml.safe_load((ROOT/'config/competition.yaml').read_text(encoding='utf-8-sig'))
        self.io.lock=threading.RLock();self.io.rings={};self.io.markers={};self.io.seq=0
        self.io.last_ar_stamp=self.io.last_ring_stamp=0
        self.io.now=lambda:10.
        self.io.aim_since=9.
        self.io.aim_stamp=0;self.io.aim_name='moving'
        self.io.target_id=7
    def header(self,at=100.,frame='usb_cam'):return NS(stamp=Stamp(at),frame_id=frame)
    def objects(self,at=100.,data=None):
        return NS(header=self.header(at),objects=NS(data=[113,100,100,1,0,0,0,1,0,200,100,1] if data is None else data))
    def ar(self,at=100.,key=7,frame='usb_cam',x=.05,y=.2,z=1.5):
        marker=NS(id=key,pose=NS(header=self.header(at,frame),pose=NS(position=NS(x=x,y=y,z=z))))
        return NS(header=self.header(at),markers=[marker])
    def test_stamped_ring_receives_center(self):
        self.io.objects_cb(self.objects());self.assertEqual(self.io.rings[113].x,250.)
    def test_ring_repeated_source_stamp_not_new_sample(self):
        self.io.objects_cb(self.objects());seq=self.io.seq
        self.io.objects_cb(self.objects());self.assertEqual(seq,self.io.seq)
    def test_ring_old_stamp_does_not_replace_newer(self):
        self.io.objects_cb(self.objects());seq=self.io.seq
        self.io.objects_cb(self.objects(99.9));self.assertEqual(seq,self.io.seq)
    def test_ring_stale_source_is_cleared(self):
        self.io.objects_cb(self.objects());self.io.objects_cb(self.objects(98.));self.assertEqual(self.io.rings,{})
    def test_ring_missing_is_cleared(self):
        self.io.objects_cb(self.objects(99.9));self.io.objects_cb(self.objects(data=[]));self.assertEqual(self.io.rings,{})
    def test_ar_repeated_stamp_not_new_sample(self):
        self.io.ar_cb(self.ar());seq=self.io.seq
        self.io.ar_cb(self.ar());self.assertEqual(seq,self.io.seq)
    def test_ar_missing_current_frame_clears(self):
        self.io.ar_cb(self.ar(99.9));self.io.ar_cb(NS(header=self.header(),markers=[]));self.assertEqual(self.io.markers,{})
    def test_wrong_camera_frame_rejected(self):
        self.io.ar_cb(self.ar(frame='base_link'));self.assertEqual(self.io.markers,{})
    def test_behind_camera_rejected(self):
        self.io.ar_cb(self.ar(z=-1));self.assertEqual(self.io.markers,{})
    def test_ar_nan_rejected(self):
        self.io.ar_cb(self.ar(x=float('nan')));self.assertEqual(self.io.markers,{})
    def test_wrong_target_not_returned(self):
        self.io.ar_cb(self.ar(key=8));self.assertIsNone(self.io.observation('moving',7))
    def test_observation_before_aim_rejected(self):
        self.io.markers[7]=Observation(7,.05,.2,1.5,8.,1)
        self.assertIsNone(self.io.observation('moving',7))
    def test_nan_odometry_not_stationary(self):
        self.io.odom=NS(twist=NS(twist=Twist()));self.io.odom.twist.twist.angular.z=float('nan')
        self.io.odom_at=10.
        self.assertFalse(self.io.stationary())
    def setup_fire(self,obs=None):
        self.io.vel_pub=Publisher();self.io.guard=lambda:None
        self.io.stop=lambda:None
        self.io.stationary=lambda:True
        self.io.markers[7]=obs or Observation(7,.05,.2,1.5,10.,1)
        self.calls=[]
        self.io.bridge = dict(dry_run=False, active=True, estop=False, mode='manual',
                              armed=False, busy=False, fault=None, pulse_id=None, pulse_result='idle')
        self.io.bridge_at = self.io.now()
        self.io.events = []
        self.io.state = lambda name, **details: self.io.events.append(dict(state=name, **details))
        def call(service,*args):
            self.calls.append(service)
            if service.startswith('/competition/fire'):
                self.io.bridge.update(armed=True, pulse_id='test-pulse', pulse_result='completed', busy=False)
                return NS(success=True, message=json.dumps(dict(pulse_id='test-pulse')))
            return NS(success=True,message='ok')
        self.io.call=call
    def test_latest_aligned_target_can_request_one_pulse(self):
        self.setup_fire();self.io.fire('moving')
        self.assertEqual(self.calls,['/competition/arm','/competition/fire_moving'])
    def test_target_lost_after_gate_never_sends_fire(self):
        self.setup_fire();self.io.markers.clear()
        with self.assertRaises(MissionFault):self.io.fire('moving')
        self.assertFalse(any(s.startswith('/competition/fire') for s in self.calls))
    def test_target_moved_after_gate_never_sends_fire(self):
        self.setup_fire(Observation(7,.50,.2,1.5,10.,1))
        with self.assertRaises(MissionFault):self.io.fire('moving')
        self.assertFalse(any(s.startswith('/competition/fire') for s in self.calls))
    def test_stale_after_gate_never_sends_fire(self):
        self.setup_fire(Observation(7,.05,.2,1.5,9.7,1))
        with self.assertRaises(MissionFault):self.io.fire('moving')
        self.assertFalse(any(s.startswith('/competition/fire') for s in self.calls))
    def test_motion_after_gate_never_sends_fire(self):
        self.setup_fire();self.io.stationary=lambda:False
        with self.assertRaises(MissionFault):self.io.fire('moving')
        self.assertFalse(any(s.startswith('/competition/fire') for s in self.calls))
    def test_hung_rpc_has_wall_deadline_and_no_duplicate(self):
        self.io.now=time.monotonic
        requests=[]
        def slow(*args):requests.append(1);time.sleep(.2);return NS(success=True)
        self.rospy.ServiceProxy=lambda *args:slow
        started=time.monotonic()
        with self.assertRaisesRegex(MissionFault,'timed out'):
            self.module.RosIO.call(self.io,'/test',object,.05,lambda:None)
        self.assertLess(time.monotonic()-started,.15)
        time.sleep(.2)
        self.assertEqual(requests,[1])


if __name__=='__main__':unittest.main()
