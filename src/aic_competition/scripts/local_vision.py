#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import threading
import time
import rospy
import rospkg
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse
from aic_competition.vision import CardMatcher


class LocalVision:
    def __init__(self):
        root = rospkg.RosPack().get_path('aic_competition')
        self.matcher = CardMatcher(rospy.get_param('~cards_dir', root + '/assets/cards'))
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.frame, self.seq, self.received = None, 0, 0
        self.last_stamp = 0
        self.timeout = rospy.get_param('~timeout', 15.0)
        self.stable_frames = rospy.get_param('~stable_frames', 3)
        self.max_frame_age = rospy.get_param('~max_frame_age', .6)
        self.required_span = rospy.get_param('~stable_span', .4)
        self.topic = rospy.get_param('/competition/config/topics/image')
        rospy.Subscriber(self.topic, Image, self.image, queue_size=1, buff_size=2**24)
        rospy.Service('/competition/recognize', Trigger, self.recognize)

    def image(self, msg):
        if msg.header.stamp.to_sec() <= 0:
            return
        age = (rospy.Time.now()-msg.header.stamp).to_sec()
        if age < -.1 or age > self.max_frame_age:
            return
        with self.lock:
            stamp = msg.header.stamp.to_nsec()
            if stamp <= self.last_stamp:
                return
            self.last_stamp = stamp
            self.frame, self.received = msg, time.monotonic()
            self.seq += 1

    def recognize(self, _):
        if not self.request_lock.acquire(False):
            return TriggerResponse(False, 'recognition already in progress')
        try:
            started = time.monotonic()
            end = started + self.timeout
            with self.lock:
                last_seq = self.seq
            previous, count, first, previous_frame_at = None, 0, None, None
            diagnostics = {'reason': 'no fresh camera frame'}
            while time.monotonic() < end and not rospy.is_shutdown():
                with self.lock:
                    msg, seq, received = self.frame, self.seq, self.received
                if seq == last_seq or msg is None or received < started or time.monotonic()-received > self.max_frame_age:
                    time.sleep(.02)
                    continue
                last_seq = seq
                try:
                    image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                    result, diagnostics = self.matcher.match(image)
                except Exception as e:
                    result, diagnostics = None, {'reason': str(e)}
                # Inference may itself outlive the request deadline; never return a late success.
                if time.monotonic() >= end:
                    break
                if result is None:
                    previous, count, first, previous_frame_at = None, 0, None, None
                    continue
                key = (result['math_answer'], result['fruit_id'])
                contiguous = previous_frame_at is not None and received-previous_frame_at < 1.5
                if key == previous and contiguous:
                    count += 1
                else:
                    previous, count, first = key, 1, received
                previous_frame_at = received
                if count >= self.stable_frames and received-first >= self.required_span:
                    out = dict(math_answer=key[0], fruit_id=key[1], template=result['file'],
                               frames=count, text_inliers=result['text_inliers'], fruit_inliers=result['fruit_inliers'])
                    rospy.loginfo('Card recognized: %s', json.dumps(out, ensure_ascii=False))
                    return TriggerResponse(True, json.dumps(out, ensure_ascii=False))
            return TriggerResponse(False, 'recognition timeout: ' + json.dumps(diagnostics, ensure_ascii=False))
        finally:
            self.request_lock.release()


if __name__ == '__main__':
    rospy.init_node('competition_local_vision')
    LocalVision()
    rospy.spin()
