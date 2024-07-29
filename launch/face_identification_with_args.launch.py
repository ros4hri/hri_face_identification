# Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from launch_ros.event_handlers import OnStateTransition
from lifecycle_msgs.msg import Transition

import ament_index_python as aip


def generate_launch_description():
    pkg_path = aip.get_package_share_directory('hri_face_identification')
    default_model_path = pkg_path + '/model/dlib_face_recognition_resnet_model_v1.dat'

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=[default_model_path],
        description='The absolute path to the face identification model')

    match_thresh_arg = DeclareLaunchArgument(
        'match_threshold',
        default_value=['0.5'],
        description='Recognition threshold (max Euclidian distance between embeddings)')

    additional_face_database_paths_arg = DeclareLaunchArgument(
        'additional_face_database_paths',
        default_value=['[""]'],
        description='List of absolute paths to the additional faces databases')

    persistent_face_database_path_arg = DeclareLaunchArgument(
        'persistent_face_database_path',
        default_value=['/tmp/faces_db'],
        description='Path to the faces database where all known faces are stored')

    can_learn_new_faces_arg = DeclareLaunchArgument(
        'can_learn_new_faces',
        default_value=['True'],
        description='Whether or not unknown faces will be added to the database')

    identify_all_faces_arg = DeclareLaunchArgument(
        'identify_all_faces',
        default_value=['False'],
        description='Whether or not faces that are already tracked are '
                    're-identified at every frame '
                    '(more accurate, but slower)')

    processing_rate_arg = DeclareLaunchArgument(
        'processing_rate',
        default_value=['10.0'],
        description='Best-effort face identification rate (Hz)')

    face_identification_node = LifecycleNode(
        package='hri_face_identification',
        executable='hri_face_identification',
        output='both',
        emulate_tty=True,
        namespace='',
        name='hri_face_identification',
        parameters=[{'model_path':
                     LaunchConfiguration('model_path'),
                     'match_threshold':
                     LaunchConfiguration('match_threshold'),
                     'additional_face_database_paths':
                     LaunchConfiguration('additional_face_database_paths'),
                     'persistent_face_database_path':
                     LaunchConfiguration('persistent_face_database_path'),
                     'can_learn_new_faces':
                     LaunchConfiguration('can_learn_new_faces'),
                     'identify_all_faces':
                     LaunchConfiguration('identify_all_faces'),
                     'processing_rate':
                     LaunchConfiguration('processing_rate')}])

    configure_event = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(face_identification_node),
        transition_id=Transition.TRANSITION_CONFIGURE))

    activate_event = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=face_identification_node, goal_state='inactive',
        entities=[EmitEvent(event=ChangeState(
            lifecycle_node_matcher=matches_action(face_identification_node),
            transition_id=Transition.TRANSITION_ACTIVATE))]))

    return LaunchDescription([
        model_path_arg,
        match_thresh_arg,
        additional_face_database_paths_arg,
        persistent_face_database_path_arg,
        can_learn_new_faces_arg,
        identify_all_faces_arg,
        processing_rate_arg,
        face_identification_node,
        configure_event,
        activate_event])
