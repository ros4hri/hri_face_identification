// Copyright (c) 2023 PAL Robotics S.L. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HRI_FACE_IDENTIFICATION__NODE_FACE_IDENTIFICATION_HPP_
#define HRI_FACE_IDENTIFICATION__NODE_FACE_IDENTIFICATION_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "hri/face.hpp"
#include "hri/hri.hpp"
#include "hri/types.hpp"
#include "hri_msgs/msg/ids_match.hpp"
#include "privacy_msgs/msg/personal_data.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include "hri_face_identification/face_recognition.hpp"

namespace hri_face_identification
{
using LifecycleCallbackReturn =
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class NodeFaceIdentification : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit NodeFaceIdentification(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~NodeFaceIdentification();

  LifecycleCallbackReturn on_configure(const rclcpp_lifecycle::State &);
  LifecycleCallbackReturn on_cleanup(const rclcpp_lifecycle::State &);
  LifecycleCallbackReturn on_activate(const rclcpp_lifecycle::State &);
  LifecycleCallbackReturn on_deactivate(const rclcpp_lifecycle::State &);
  LifecycleCallbackReturn on_shutdown(const rclcpp_lifecycle::State &);

private:
  void internal_cleanup();
  void internal_deactivate();
  void publishDiagnostics();
  void processFaces();

  std::unique_ptr<FaceRecognition> face_recognition_;
  std::shared_ptr<hri::HRIListener> hri_listener_;
  std::string persistent_face_database_path_;
  std::vector<std::string> loaded_face_database_paths_;
  bool can_learn_new_faces_;
  bool identify_all_faces_;
  double processing_rate_;
  rclcpp::Publisher<hri_msgs::msg::IdsMatch>::SharedPtr candidate_matches_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
  rclcpp::Publisher<privacy_msgs::msg::PersonalData>::SharedPtr privacy_pub_;
  std::shared_ptr<rclcpp::TimerBase> diagnostics_timer_;
  std::shared_ptr<rclcpp::TimerBase> process_images_timer_;
  std::map<Id, hri::FacePtr> tracked_faces_;
  // mapping between a face_id and all the possible recognised person_id
  // (with their confidence level) for that face.
  std::map<Id, std::map<Id, float>> face_persons_map_;
};

}  // namespace hri_face_identification

#endif  // HRI_FACE_IDENTIFICATION__NODE_FACE_IDENTIFICATION_HPP_
