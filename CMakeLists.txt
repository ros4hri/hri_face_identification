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

cmake_minimum_required(VERSION 3.0.2)
project(hri_face_identification)

find_package(catkin REQUIRED COMPONENTS
  hri_msgs
  image_transport
  hri
  roscpp
)

find_package(dlib REQUIRED)
find_package(BLAS REQUIRED)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

catkin_package(
  CATKIN_DEPENDS hri_msgs image_transport hri roscpp
  DEPENDS dlib
)

###########
## Build ##
###########

include_directories(
    SYSTEM ${catkin_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/identify.cpp src/face_recognition.cpp)
set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME identify PREFIX "" CXX_STANDARD 11)

target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
   dlib::dlib
   ${BLAS_LIBRARIES}
)

add_executable(batch_process src/batch_process.cpp src/face_recognition.cpp)
set_property(TARGET batch_process PROPERTY CXX_STANDARD 17)
target_link_libraries(batch_process
   ${catkin_LIBRARIES}
   dlib::dlib
   ${BLAS_LIBRARIES}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
install(TARGETS ${PROJECT_NAME}_node batch_process
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
   share/dlib_face_recognition_resnet_model_v1.dat
   share/test_faces_db.json
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/share
)

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

###########################################################
# Tests

if(CATKIN_ENABLE_TESTING)

find_package(catkin REQUIRED COMPONENTS
  hri_msgs
  image_transport
  hri
  roscpp
  rosbag
  diagnostic_msgs
  diagnostic_updater
)

include_directories(
    src
)

find_package(rostest REQUIRED)

add_rostest_gtest(test_hri_face_id test/test_face_id.test test/test_face_id.cpp src/face_recognition.cpp)
target_include_directories(test_hri_face_id PRIVATE ${rostest_INCLUDE_DIRS})
target_link_libraries(test_hri_face_id
   ${catkin_LIBRARIES}
   dlib::dlib
   ${BLAS_LIBRARIES}
)


endif() # CATKIN_ENABLE_TESTING



