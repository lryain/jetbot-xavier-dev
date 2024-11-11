#!/usr/bin/env python3

import threading

import numpy as np

import rospy
import tf
from geometry_msgs.msg import PointStamped, Point, Vector3
from video_analyzer.msg import VideoAnalysis
from audio_analyzer.msg import AudioAnalysis
from person_identification.msg import PersonNames, PersonName

import person_identification


PERSON_POSE_NOSE_INDEX = 0


def calculate_distance(a, b):
    return np.linalg.norm(a - b)


def calculate_angle(a, b):
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


class FaceDescriptorData:
    def __init__(self, descriptor, position_2d, position_3d, direction):
        self.descriptor = descriptor
        self.position_2d = position_2d
        self.position_3d = position_3d
        self.direction = direction


class VoiceDescriptorData:
    def __init__(self, descriptor, direction):
        self.descriptor = descriptor
        self.direction = direction


class PersonIdentificationNode:
    def __init__(self):
        self._face_sharpness_score_threshold = rospy.get_param('~face_sharpness_score_threshold')
        self._face_descriptor_threshold = rospy.get_param('~face_descriptor_threshold')
        self._voice_descriptor_threshold = rospy.get_param('~voice_descriptor_threshold')
        self._face_voice_descriptor_threshold = rospy.get_param('~face_voice_descriptor_threshold')
        self._nose_confidence_threshold = rospy.get_param('~nose_confidence_threshold')
        self._direction_frame_id = rospy.get_param('~direction_frame_id')
        self._direction_angle_threshold_rad = rospy.get_param('~direction_angle_threshold_rad')
        self._ignore_direction_z = rospy.Rate(rospy.get_param('~ignore_direction_z'))
        self._rate = rospy.Rate(rospy.get_param('~search_frequency'))

        self._face_descriptors_by_name = {}
        self._voice_descriptors_by_name = {}
        self._face_voice_descriptors_by_name = {}
        self._load_descriptors()

        self._descriptors_lock = threading.Lock()
        self._face_descriptor_data = []
        self._voice_descriptor_data = None
        self._person_name_pub = rospy.Publisher('person_names', PersonNames, queue_size=5)

        self._tf_listener = tf.TransformListener()
        self._video_analysis_sub = rospy.Subscriber('video_analysis', VideoAnalysis, self._video_analysis_cb, queue_size=1)
        self.audio_analysis = rospy.Subscriber('audio_analysis', AudioAnalysis, self._audio_analysis_cb, queue_size=1)

    def _load_descriptors(self):
        # rospy.logdebug("000 _load_descriptors...")
        people = person_identification.load_people()
        for name, descriptors in people.items():
            rospy.logdebug("0. _load_descriptors ----------------------------> ################# name: %s ", name)
            if 'face' in descriptors:
                self._face_descriptors_by_name[name] = np.array(descriptors['face'])
                rospy.logdebug("1. 人脸_descriptors ------->  face: %s ", self._face_descriptors_by_name[name])
            if 'voice' in descriptors:
                self._voice_descriptors_by_name[name] = np.array(descriptors['voice'])
                rospy.logdebug("1. 声音_descriptors ------->  voice: %s ", self._voice_descriptors_by_name[name])
            if 'face' in descriptors and 'voice' in descriptors:
                self._face_voice_descriptors_by_name[name] = np.array(descriptors['face'] + descriptors['voice'])
                rospy.logdebug("1. 人脸和声音_descriptors ------->  face_voice: %s ", self._face_voice_descriptors_by_name[name])

    def _video_analysis_cb(self, msg):
        # rospy.logdebug("000 _video_analysis_cb...")
        if not msg.contains_3d_positions:
            rospy.logdebug('The video analysis must contain 3d positions.')
            return

        with self._descriptors_lock:
            for object in msg.objects:
                if len(object.face_descriptor) == 0 or len(object.person_pose_2d) == 0 or len(object.person_pose_3d) == 0 \
                        or len(object.person_pose_confidence) == 0 \
                        or object.person_pose_confidence[PERSON_POSE_NOSE_INDEX] < self._nose_confidence_threshold \
                        or object.face_sharpness_score < self._face_sharpness_score_threshold:
                    continue

                position_2d = object.person_pose_2d[PERSON_POSE_NOSE_INDEX]
                # rospy.logdebug("000 start _get_face_position_3d_and_direction...")
                
                position_3d, direction = self._get_face_position_3d_and_direction(object.person_pose_3d[PERSON_POSE_NOSE_INDEX], msg.header)
                if np.isfinite(direction).all():
                    self._face_descriptor_data.append(FaceDescriptorData(np.array(object.face_descriptor),
                                                                         position_2d,
                                                                         position_3d,
                                                                         direction))

    def _get_face_position_3d_and_direction(self, point, header):
        temp_in_point = PointStamped()
        temp_in_point.header = header
        temp_in_point.point.x = point.x
        temp_in_point.point.y = point.y
        temp_in_point.point.z = point.z
        # ！！这个硬是把tf中的 odas换成了camera_3d_color_frame后可以跑起来
        # rospy.logdebug("_get_face_position_3d_and_direction  ---> transformPoint( {temp_in_point} )", )

        odas_point = self._tf_listener.transformPoint(self._direction_frame_id, temp_in_point)

        position = np.array([odas_point.point.x, odas_point.point.y, odas_point.point.z])
        direction = position.copy()
        if self._ignore_direction_z:
            direction[2] = 0

        direction /= np.linalg.norm(direction)
        return position, direction

    def _audio_analysis_cb(self, msg):
        if msg.header.frame_id != self._direction_frame_id:
            rospy.logdebug(f'Invalid frame id ({msg.header.frame_id} != {self._direction_frame_id})')
            return

        if len(msg.voice_descriptor) == 0:
            rospy.logdebug("￥￥￥￥￥￥￥￥￥￥￥￥￥￥ _audio_analysis_cb ---------> len(msg.voice_descriptor) == 0")
            return
        rospy.logdebug("0. _audio_analysis_cb ---------> msg.direction_x: %s msg.direction_y: %s msg.direction_z: %s", msg.direction_x, msg.direction_y, msg.direction_z)
        voice_direction = np.array([msg.direction_x, msg.direction_y, msg.direction_z])
        rospy.logdebug("1. _audio_analysis_cb ---------> voice_direction: %s", voice_direction)
        if self._ignore_direction_z:
            voice_direction[2] = 0

        voice_direction /= np.linalg.norm(voice_direction)
        voice_direction_all = np.isfinite(voice_direction).all()
        rospy.logdebug("2. _audio_analysis_cb ---------> voice_direction: %s", voice_direction)
        voice_direction_all = True
        if voice_direction_all:
            with self._descriptors_lock:
                rospy.logdebug("3. _audio_analysis_cb ---------> VoiceDescriptorData() voice_direction: %s", voice_direction)
                self._voice_descriptor_data = VoiceDescriptorData(np.array(msg.voice_descriptor), voice_direction)
        # else:
            # rospy.logdebug("4. _audio_analysis_cb ---------> voice_direction_all: %s", voice_direction_all)

    def run(self):
        while not rospy.is_shutdown():
            with self._descriptors_lock:
                names = []
                names.extend(self._find_face_voice_descriptor_name())
                names.extend(self._find_voice_descriptor_name())
                names.extend(self._find_face_descriptor_names())

                self._face_descriptor_data.clear()
                self._voice_descriptor_data = None
            
            # if names is not None and len(names) > 0 :
                # rospy.logdebug("9. run _publish_names ---------> name is %s", names)
            
            self._publish_names(names)
            self._rate.sleep()
    
    # 人脸和声音都匹配到
    def _find_face_voice_descriptor_name(self):
        # return[]
        if self._voice_descriptor_data is None or len(self._face_descriptor_data) == 0 or \
                len(self._face_voice_descriptors_by_name) == 0:
            rospy.logdebug("0. 人脸和声音同时匹配 ---------> 人脸或者声音特征为空！")
            return []

        face_descriptor_index_angle_pairs = [(i, calculate_angle(x.direction, self._voice_descriptor_data.direction))
                                             for i, x in enumerate(self._face_descriptor_data)]
        face_descriptor_index, angle = min(face_descriptor_index_angle_pairs, key=lambda x: x[1])
        if angle > self._direction_angle_threshold_rad:
            rospy.logdebug("1. 人声角度不对！angle: %s", angle)
            return []

        face_descriptor_data = self._face_descriptor_data[face_descriptor_index]
        descriptor = np.concatenate([face_descriptor_data.descriptor, self._voice_descriptor_data.descriptor])

        name_distance_pairs = [(x[0], calculate_distance(x[1], descriptor))
                               for x in self._face_voice_descriptors_by_name.items()]
        name, distance = min(name_distance_pairs, key=lambda x: x[1])
        rospy.logdebug("2. 同时匹配到 ---------> _face_descriptor_data is name: %s", name)
        if distance <= self._face_voice_descriptor_threshold:
            # rospy.logdebug("3. 人声距离符合范围: distance %s", distance)
            self._voice_descriptor_data = None
            del self._face_descriptor_data[face_descriptor_index]
            rospy.logdebug("4. ！！！！！！！！face_and_voice 完全匹配且符合要求: name %s", name)
            return [self._create_person_name(name,
                                             'face_and_voice',
                                             position_2d=face_descriptor_data.position_2d,
                                             position_3d=face_descriptor_data.position_3d,
                                             direction=face_descriptor_data.direction)]
        else:
            rospy.logdebug("1. 人声距离太远: distance %s", distance)
            return []

    def _find_face_descriptor_names(self):
        # rospy.logdebug("_find_face_descriptor_names --------->")
        if len(self._face_descriptors_by_name) == 0:
            rospy.logdebug("姓名描述符为空，请先录入人脸，返回！")
            return []

        names = []
        # rospy.logdebug("姓名描述符不为空，开始循环 查找该 面部 face!!")
        
        for face_descriptor_data in self._face_descriptor_data:
            name_distance_pairs = [(x[0], calculate_distance(x[1], face_descriptor_data.descriptor))
                                   for x in self._face_descriptors_by_name.items()]
            name, distance = min(name_distance_pairs, key=lambda x: x[1])

            if distance <= self._face_descriptor_threshold:
                rospy.loginfo("_find_face_descriptor_names ---------> 将姓名: {%s} 添加到 names 中", name)
                names.append(self._create_person_name(name,
                                                      'face',
                                                      position_2d=face_descriptor_data.position_2d,
                                                      position_3d=face_descriptor_data.position_3d,
                                                      direction=face_descriptor_data.direction))

        return names # filter names

    def _find_voice_descriptor_name(self):
        if self._voice_descriptor_data is None or len(self._voice_descriptors_by_name) == 0:
            rospy.logdebug("1. 声音匹配 --------------> 声音人数据为空，或者没有检测到人声特征！")
            return []

        name_distance_pairs = [(x[0], calculate_distance(x[1], self._voice_descriptor_data.descriptor))
                               for x in self._voice_descriptors_by_name.items()]
        name, distance = min(name_distance_pairs, key=lambda x: x[1])
        rospy.logdebug("2. 声音匹配-----------------------------> 匹配到声音人 name: %s", name)
        if distance <= self._voice_descriptor_threshold:
            rospy.logdebug("3. 声音匹配-----------------------------> 人声音够大 name: %s", name)
            # rospy.logdebug("4. ~~~~~~~~~~ voice 完全匹配且符合要求: name %s", name)
            return [self._create_person_name(name,
                                             'voice',
                                             direction=self._voice_descriptor_data.direction)]
        else:
            rospy.logdebug("5. 声音匹配-----------------------------> 距离太远 name: %s", name)
            return []

    def _create_person_name(self, name, detection_type, position_2d=None, position_3d=None, direction=None):
        rospy.logdebug("-----------------------------> _create_person_name detection_type(%s) - name(%s)", detection_type, name)
        person_name = PersonName()
        person_name.name = name
        person_name.detection_type = detection_type
        person_name.frame_id = self._direction_frame_id
        if position_2d is not None:
            person_name.position_2d.append(position_2d)
        if position_3d is not None:
            person_name.position_3d.append(Point(x=position_3d[0], y=position_3d[1], z=position_3d[2]))
        if direction is not None:
            person_name.direction.append(Vector3(x=direction[0], y=direction[1], z=direction[2]))

        return person_name

    def _publish_names(self, names):
        # rospy.logdebug("--------------------------------> _publish_names: %s", names)
        msg = PersonNames()
        msg.names = self._filter_names(names)
        self._person_name_pub.publish(msg)

    def _filter_names(self, names):
        inserted_names = set()
        filtered_names = []

        for name in reversed(names):
            if name.name not in inserted_names:
                filtered_names.append(name)
                inserted_names.add(name.name)

        return filtered_names


def main():
    rospy.init_node('person_identification_node')
    person_identification_node = PersonIdentificationNode()
    person_identification_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
