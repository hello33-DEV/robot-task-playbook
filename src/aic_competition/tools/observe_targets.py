#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only target diagnostics for axis, ID, freshness and barrel-offset calibration."""
import math
import time
import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
from find_object_2d.msg import ObjectsStamped
from aic_competition.core import object_centers

rospy.init_node('aic_observe_targets', anonymous=True)
last = {}
def ar(msg):
    now=time.monotonic()
    for m in msg.markers:
        if now-last.get(('ar',m.id),0)<.5:
            continue
        last[('ar',m.id)]=now
        p=m.pose.pose.position
        age=(rospy.Time.now()-msg.header.stamp).to_sec()
        rospy.loginfo('AR id=%d frame=%s x=%.4f y=%.4f z=%.4f age=%.3fs',m.id,
                      m.pose.header.frame_id or msg.header.frame_id,p.x,p.y,p.z,age)
def ring(msg):
    try:
        centers=object_centers(msg.objects.data)
    except ValueError as e:
        rospy.logwarn(str(e));return
    for key,center in centers.items():
        now=time.monotonic()
        if now-last.get(('ring',key),0)<.5:
            continue
        last[('ring',key)]=now
        rospy.loginfo('Ring id=%d center=(%.2f,%.2f) age=%.3fs',key,*center,
                      (rospy.Time.now()-msg.header.stamp).to_sec())
rospy.Subscriber('/ar_pose_marker',AlvarMarkers,ar,queue_size=1)
rospy.Subscriber('/objectsStamped',ObjectsStamped,ring,queue_size=1)
rospy.loginfo('Read only: no navigation, velocity, arm or firing commands are sent.')
rospy.spin()
