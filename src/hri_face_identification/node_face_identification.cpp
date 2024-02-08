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

#include "hri_face_identification/node_face_identification.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <functional>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "ament_index_cpp/get_resource.hpp"
#include "ament_index_cpp/get_resources.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_updater/diagnostic_status_wrapper.hpp"
#include "dlib/serialize.h"
#include "hri/face.hpp"
#include "hri/hri.hpp"
#include "hri_msgs/msg/ids_match.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "opencv2/highgui.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include "hri_face_identification/face_recognition.hpp"

namespace hri_face_identification
{
NodeFaceIdentification::NodeFaceIdentification(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("hri_face_identification", "", options)
{
  auto descriptor = rcl_interfaces::msg::ParameterDescriptor{};

  auto pkg_path = ament_index_cpp::get_package_share_directory("hri_face_identification");
  auto default_model_path = pkg_path + "/model/dlib_face_recognition_resnet_model_v1.dat";
  descriptor.description = "Absolute path to the face identification model";
  this->declare_parameter("model_path", default_model_path, descriptor);

  descriptor.description = "Recognition threshold (max Euclidian distance between embeddings)";
  this->declare_parameter("match_threshold", 0.5, descriptor);

  std::vector<std::string> default_face_database_paths;
  auto faces_db_resources = ament_index_cpp::get_resources("faces_database");
  for (const auto & [pkg, share_path] : faces_db_resources) {
    std::string db_file_name;
    if (ament_index_cpp::get_resource("faces_database", pkg, db_file_name)) {
      auto db_path =
        std::filesystem::path(ament_index_cpp::get_package_share_directory(pkg)) / db_file_name;
      default_face_database_paths.emplace_back(db_path.string());
    }
  }
  descriptor.description = "List of absolute paths to the known faces databases";
  this->declare_parameter("face_database_paths", default_face_database_paths, descriptor);

  descriptor.description = "Whether or not unknown faces will be added to the database";
  this->declare_parameter("can_learn_new_faces", true, descriptor);

  descriptor.description = "Whether or not faces that are already tracked are re-identified at "
    "every frame (more accurate, but slower)";
  this->declare_parameter("identify_all_faces", false, descriptor);

  descriptor.description = "Best-effort face identification rate (Hz)";
  this->declare_parameter("processing_rate", 10., descriptor);

  RCLCPP_INFO(this->get_logger(), "State: Unconfigured");
}

LifecycleCallbackReturn NodeFaceIdentification::on_configure(const rclcpp_lifecycle::State &)
{
  face_database_paths_param_ = this->get_parameter("face_database_paths");
  can_learn_new_faces_ = this->get_parameter("can_learn_new_faces").as_bool();
  identify_all_faces_ = this->get_parameter("identify_all_faces").as_bool();
  processing_rate_ = this->get_parameter("processing_rate").as_double();

  try {
    face_recognition_ = std::make_unique<FaceRecognition>(
      this->get_parameter("model_path").as_string(),
      this->get_parameter("match_threshold").as_double());
  } catch (dlib::serialization_error & e) {
    RCLCPP_ERROR_STREAM(this->get_logger(), "Could not load the model: " << e.what());
    return LifecycleCallbackReturn::FAILURE;
  }

  for (const auto & path : face_database_paths_param_.as_string_array()) {
    if (face_recognition_->loadFaceDB(path)) {
      RCLCPP_INFO_STREAM(this->get_logger(), "Face database correctly loaded from " << path);
    } else {
      RCLCPP_WARN_STREAM(
        this->get_logger(),
        "Unable to load face database from " << path << ". Starting with no known faces.");
    }
  }

  RCLCPP_INFO(this->get_logger(), "State: Inactive");
  return LifecycleCallbackReturn::SUCCESS;
}

LifecycleCallbackReturn NodeFaceIdentification::on_cleanup(const rclcpp_lifecycle::State &)
{
  internal_cleanup();
  RCLCPP_INFO(this->get_logger(), "State: Unconfigured");
  return LifecycleCallbackReturn::SUCCESS;
}

LifecycleCallbackReturn NodeFaceIdentification::on_activate(const rclcpp_lifecycle::State &)
{
  candidate_matches_pub_ = this->create_publisher<hri_msgs::msg::IdsMatch>(
    "/humans/candidate_matches", 10);
  diagnostics_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "/diagnostics", 1);

  hri_listener_ = hri::HRIListener::create(shared_from_this());
  hri_listener_->onFace([&](const hri::FacePtr & face) {tracked_faces_[face->id()] = face;});

  process_images_timer_ = rclcpp::create_timer(
    this, this->get_clock(), std::chrono::nanoseconds(static_cast<int>(1e9 / processing_rate_)),
    std::bind(&NodeFaceIdentification::processFaces, this));
  diagnostics_timer_ = rclcpp::create_timer(
    this, this->get_clock(), std::chrono::seconds(1),
    std::bind(&NodeFaceIdentification::publishDiagnostics, this));

  RCLCPP_INFO_STREAM(
    this->get_logger(),
    this->get_name() << "running. Waiting for faces to be published on /humans/faces/.... "
      "Results of face identification will be published on /humans/candidate_matches.");
  RCLCPP_INFO(this->get_logger(), "State: Active");
  return LifecycleCallbackReturn::SUCCESS;
}

LifecycleCallbackReturn NodeFaceIdentification::on_deactivate(const rclcpp_lifecycle::State &)
{
  internal_deactivate();
  RCLCPP_INFO(this->get_logger(), "State: Inactive");
  return LifecycleCallbackReturn::SUCCESS;
}

LifecycleCallbackReturn NodeFaceIdentification::on_shutdown(const rclcpp_lifecycle::State & state)
{
  if (state.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
    internal_deactivate();
  }
  internal_cleanup();
  RCLCPP_INFO(this->get_logger(), "State: Finalized");
  return LifecycleCallbackReturn::SUCCESS;
}

void NodeFaceIdentification::internal_deactivate()
{
  hri_listener_.reset();
  diagnostics_pub_.reset();
  candidate_matches_pub_.reset();
  face_persons_map_.clear();
  tracked_faces_.clear();
}

void NodeFaceIdentification::internal_cleanup()
{
  const auto & path = face_database_paths_param_.as_string_array()[0];
  face_recognition_->storeFaceDB(path);
  RCLCPP_INFO_STREAM(this->get_logger(), "Face database correctly saved to " << path << std::endl);
  face_recognition_.reset();
}

void NodeFaceIdentification::processFaces()
{
  std::vector<Id> faces_to_remove;

  for (const auto & [face_id, face] : tracked_faces_) {
    if (auto face_aligned = face->aligned()) {
      RCLCPP_DEBUG_STREAM(this->get_logger(), "Got face " << face_id);
      std::map<Id, float> results;

      if (identify_all_faces_ || !face_persons_map_.count(face_id)) {
        RCLCPP_INFO(this->get_logger(), "Trying to identify the face...");
        // note that this might return more than one match!
        // each match has an associated confidence level
        bool new_person_created;
        results = face_recognition_->getAllMatches(
          *face_aligned, can_learn_new_faces_, new_person_created);

        if (!results.empty()) {
          face_persons_map_[face_id] = results;

          if (new_person_created) {
            RCLCPP_INFO_STREAM(
              this->get_logger(),
              "New person detected; will be identified as <" << results.begin()->first << ">");
          } else if (results.size() == 1U) {
            RCLCPP_INFO_STREAM(
              this->get_logger(),
              "Found a match with person " << results.begin()->first << " (confidence: " <<
                results.begin()->second);
          } else {
            RCLCPP_INFO(this->get_logger(), "Found more than one possible match:");
            for (const auto & result : results) {
              RCLCPP_INFO_STREAM(
                this->get_logger(), "  - " << result.first << " (c=" << result.second << ")");
            }
          }
        }
      }

      for (const auto & [person_id, confidence] : face_persons_map_[face_id]) {
        hri_msgs::msg::IdsMatch match;
        match.id1 = person_id;
        match.id1_type = hri_msgs::msg::IdsMatch::PERSON;
        match.confidence = confidence;
        match.id2 = face_id;
        match.id2_type = hri_msgs::msg::IdsMatch::FACE;

        candidate_matches_pub_->publish(match);
      }
    } else if (!face->valid()) {
      // the face does not exist anymore!
      faces_to_remove.push_back(face_id);

      // for all the person id previously associated to this face,
      // publish a 'match' with confidence = 0 to dis-associate them.
      for (const auto & [person_id, confidence] : face_persons_map_[face_id]) {
        RCLCPP_INFO_STREAM(
          this->get_logger(),
          "Face " << face_id << " not tracked anymore. Dis-associating from person " << person_id);

        hri_msgs::msg::IdsMatch match;
        match.id1 = person_id;
        match.id1_type = hri_msgs::msg::IdsMatch::PERSON;
        match.confidence = 0.0;
        match.id2 = face_id;
        match.id2_type = hri_msgs::msg::IdsMatch::FACE;

        candidate_matches_pub_->publish(match);
      }
    }
  }

  for (const auto & id : faces_to_remove) {
    tracked_faces_.erase(id);
  }
}

void NodeFaceIdentification::publishDiagnostics()
{
  diagnostic_updater::DiagnosticStatusWrapper status;
  status.name = "Social perception: Face analysis: Identification";
  status.hardware_id = "none";
  status.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "OK");
  status.add("Package name", "hri_face_identification");
  status.add("Face database paths", face_database_paths_param_.value_to_string());
  status.add("Currently detected faces", tracked_faces_.size());

  auto face_recognition_diagnostics = face_recognition_->getDiagnostics();
  status.add("Known faces", face_recognition_diagnostics.known_faces);
  status.add("Last recognized face ID", face_recognition_diagnostics.last_face_id);

  diagnostic_msgs::msg::DiagnosticArray msg;
  msg.header.stamp = this->get_clock()->now();
  msg.status.push_back(status);
  diagnostics_pub_->publish(msg);
}

}  // namespace hri_face_identification
