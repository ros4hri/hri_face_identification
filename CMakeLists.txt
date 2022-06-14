# Copyright (c) 2022 PAL Robotics S.L. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from this
#  software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/share
)

install(DIRECTORY test-data
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



