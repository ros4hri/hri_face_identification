cmake_minimum_required(VERSION 3.0.2)
project(hri_face_identification)

find_package(catkin REQUIRED)

catkin_package(
    CATKIN_DEPENDS 
    sensor_msgs
    hri_msgs
    cv_bridge)

catkin_python_setup()

catkin_install_python(PROGRAMS nodes/identify
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

#install(DIRECTORY
#  launch
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
#)

