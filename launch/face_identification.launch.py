# Copyright (c) 2023 PAL Robotics S.L. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch_ros.actions import LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from launch_ros.event_handlers import OnStateTransition
from lifecycle_msgs.msg import Transition
import os


def generate_launch_description():
    test_faces_db_path = os.path.join(
        get_package_share_directory('hri_face_identification'), 'model', 'data', 'faces_db.json')

    face_identification_node = LifecycleNode(
        package='hri_face_identification', executable='hri_face_identification', namespace='',
        name='hri_face_identification',
        parameters=[{
            'face_database_paths': ['/home/ros/.pal/face_db/face_db.json', test_faces_db_path]}])

    configure_event = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(face_identification_node),
        transition_id=Transition.TRANSITION_CONFIGURE))

    activate_event = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=face_identification_node, goal_state='inactive',
        entities=[EmitEvent(event=ChangeState(
            lifecycle_node_matcher=matches_action(face_identification_node),
            transition_id=Transition.TRANSITION_ACTIVATE))]))

    return LaunchDescription([
        face_identification_node,
        configure_event,
        activate_event])
