#!/usr/bin/env python3

import threading

import numpy as np

import rospy
from hbba_lite.srv import SetThrottlingFilterState
from video_analyzer.msg import VideoAnalysis

import person_identification


INACTIVE_SLEEP_DURATION = 0.1


class CaptureFaceNode:
    def __init__(self):
        self._name = rospy.get_param('~name')
        self._mean_size = rospy.get_param('~mean_size')
        self._face_sharpness_score_threshold = rospy.get_param('~face_sharpness_score_threshold')

        self._descriptors_lock = threading.Lock()
        self._descriptors = []
        self._video_analysis_sub = rospy.Subscriber('video_analysis', VideoAnalysis, self._video_analysis_cb, queue_size=1)

    def _video_analysis_cb(self, msg):
        face_object = None
        for object in msg.objects:
            rospy.logdebug("1.0 _video_analysis_cb ----->  loop msg.objects...")
            if len(object.face_descriptor) > 0 and face_object is not None:
                rospy.logdebug("1.1 ------> 已经找到一个人脸，一次只能一个！")
                rospy.logwarn('Only one face must be present in the image.')
            elif len(object.face_descriptor) > 0:
                rospy.logdebug("1.2 ------> 找到一个人脸！")
                face_object = object

        if face_object is not None and face_object.face_sharpness_score >= self._face_sharpness_score_threshold:
            rospy.logdebug("1.3 ------> 人脸数据ok，添加到_descriptors中！")
            with self._descriptors_lock:
                self._descriptors.append(face_object.face_descriptor)
        else:
            rospy.logdebug("1.3 ------> 人脸数据质量差:( %s)，抛弃！", face_object.face_sharpness_score)
    
    def run(self):
        self.enable_video_analyzer()
        rospy.logdebug("4.1 CaptureFaceNode ----->   run.....enable_video_analyzer")
        while not rospy.is_shutdown():
            with self._descriptors_lock:
                size = len(self._descriptors)
            
            if size == self._mean_size:
                rospy.logdebug("4.2 CaptureFaceNode ----->   _save_new_descriptor.....")
                self._video_analysis_sub.unregister()
                self._save_new_descriptor()
                return
            else:
                rospy.logdebug("4.1 CaptureFaceNode ----->   人脸描述符数据为空！")
                print("4.2.1  size != self._mean_size size: ", size, " _mean_size: ", self._mean_size)
                rospy.sleep(INACTIVE_SLEEP_DURATION)

    def enable_video_analyzer(self):
        VIDEO_ANALYZER_FILTER_STATE_SERVICE = 'video_analyzer/image_raw/filter_state'

        rospy.wait_for_service(VIDEO_ANALYZER_FILTER_STATE_SERVICE)
        filter_state = rospy.ServiceProxy(VIDEO_ANALYZER_FILTER_STATE_SERVICE, SetThrottlingFilterState)
        filter_state(is_filtering_all_messages=False, rate=1)

    def _save_new_descriptor(self):
        with self._descriptors_lock:
            descriptor = np.array(self._descriptors).mean(axis=0).tolist()
            people = person_identification.load_people()
            person_identification.add_person(people, self._name, {'face': descriptor})
            person_identification.save_people(people)
            rospy.logdebug("4.3 CaptureFaceNode ---------------------->   _save_new_descriptor..... people: %s", people)

        rospy.loginfo('*********************** FINISHED ***********************')


def main():
    rospy.init_node('capture_face_node')
    capture_face_node = CaptureFaceNode()
    capture_face_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
