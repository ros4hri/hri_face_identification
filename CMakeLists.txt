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

cmake_minimum_required(VERSION 3.8)
project(hri_face_identification)

# Global flags
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(dlib REQUIRED)
find_package(BLAS REQUIRED)

# Targets
ament_auto_add_library(${PROJECT_NAME}_lib
  src/hri_face_identification/node_face_identification.cpp
  src/hri_face_identification/face_recognition.cpp
)
target_link_libraries(${PROJECT_NAME}_lib
  dlib::dlib
  ${BLAS_LIBRARIES}
)

ament_auto_add_executable(${PROJECT_NAME} src/main.cpp)

ament_auto_add_executable(db_record src/db_record.cpp)
target_link_libraries(db_record fmt::fmt)

ament_auto_add_executable(db_process src/db_process.cpp)

# Additional install
# this will install also the faces database model/data/faces_db.json
install(
  DIRECTORY launch model config
  DESTINATION share/${PROJECT_NAME}
)
# this will register faces database resource
ament_index_register_resource("faces_database" CONTENT model/data/faces_db.json)
# register the default configuration file
ament_index_register_resource("pal_configuration.hri_face_identification" CONTENT config/00-defaults.yml)

if(pal_module_cmake_FOUND)
  pal_register_modules(
    module/hri_face_identification_module.yaml
  )
endif()

# Testing
if(BUILD_TESTING)
  ament_auto_find_test_dependencies()
  ament_lint_auto_find_test_dependencies()

  ament_auto_add_gtest(${PROJECT_NAME}_test
    test/test_face_recognition.cpp
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test
    TIMEOUT 120
  )
endif()

ament_auto_package()
