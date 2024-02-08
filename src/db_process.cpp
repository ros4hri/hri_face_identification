// Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "dlib/matrix.h"
#include "dlib/opencv.h"
#include "nlohmann/json.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "tclap/ArgException.h"
#include "tclap/CmdLine.h"
#include "tclap/ValueArg.h"

#include "hri_face_identification/face_recognition.hpp"

inline std::string generate_hash_id(std::string name, const int len = 5)
{
  static const std::array<char, 26> alphanum{{
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}};
  std::string tmp_s;
  tmp_s.reserve(len);

  auto hash = std::hash<std::string>{}(name);

  for (int i = 0; i < len; ++i) {
    tmp_s += alphanum[hash % alphanum.size()];
    hash /= alphanum.size();
  }

  return tmp_s;
}

bool is_image(std::filesystem::path file_path)
{
  static const std::vector<std::string> kExtensions({".jpg", ".jpeg", ".png", ".JPG", ".JPEG"});
  return std::find(
    kExtensions.begin(), kExtensions.end(), file_path.extension()) != kExtensions.end();
}

int main(int argc, char ** argv)
{
  std::filesystem::path load_path;
  std::filesystem::path output_path;
  std::filesystem::path embeddings_path;

  try {
    TCLAP::CmdLine cmd("Face recordings processor into embeddings database", ' ');

    TCLAP::ValueArg<std::string> data_arg(
      "d", "data", "Base path where the face images are loaded from.", false, "/tmp/face_dataset",
      "string");
    cmd.add(data_arg);

    TCLAP::ValueArg<std::string> output_arg(
      "o", "output", "Path where the output face database is saved.", false,
      "/tmp/face_dataset/face_db.json", "string");
    cmd.add(output_arg);

    TCLAP::ValueArg<std::string> embeddings_arg(
      "e", "embeddings", "Path where the raw face embeddings is saved (none if empty)", false,
      "", "string");
    cmd.add(embeddings_arg);

    cmd.parse(argc, argv);
    load_path = data_arg.getValue();
    output_path = output_arg.getValue();
    embeddings_path = embeddings_arg.getValue();
  } catch (TCLAP::ArgException & e) {
    std::cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
    return 1;
  }

  std::cout << "Processing images in " << load_path << std::endl;

  auto pkg_path = ament_index_cpp::get_package_share_directory("hri_face_identification");
  auto default_model_path = pkg_path + "/model/dlib_face_recognition_resnet_model_v1.dat";
  // threshold is irrelevant as we are will only extract embeddings, not match them
  hri_face_identification::FaceRecognition face_recognition(default_model_path, 0.);

  // person{scene{image_paths}}
  std::map<std::string, std::map<std::string, std::set<std::filesystem::path>>> ordered_paths;

  for (const auto & dirEntry : std::filesystem::recursive_directory_iterator(load_path)) {
    if (dirEntry.is_regular_file() && is_image(dirEntry.path())) {
      auto relative_path = dirEntry.path().lexically_relative(load_path);
      std::string current_person = *(relative_path.begin());
      std::string current_scene = *(++(relative_path.begin()));
      ordered_paths.try_emplace(current_person);
      ordered_paths[current_person].try_emplace(current_scene);
      ordered_paths[current_person][current_scene].insert(dirEntry.path());
    }
  }

  std::map<hri::ID, std::vector<hri_face_identification::Features>> embeddings;
  std::map<std::string, std::string> person_to_id;
  std::ofstream raw_embeddings;
  if (!embeddings_path.empty()) {
    raw_embeddings.open(embeddings_path.string(), std::ios::trunc);
  }

  for (const auto & [person, scenes] : ordered_paths) {
    // use reproducible hash to anonymize
    std::string id = generate_hash_id(person);
    person_to_id[person] = id;
    std::cout << "Processing " << person << "..." << std::endl;
    embeddings.emplace(id, std::vector<hri_face_identification::Features>());
    for (const auto & [scene, paths] : scenes) {
      dlib::matrix<float> features_matrix;
      int index = 0;
      for (const auto & path : paths) {
        cv::Mat cv_face = cv::imread(path);
        cv::Mat resized_face;
        cv::resize(cv_face, resized_face, {IMG_SIZE, IMG_SIZE}, 0, 0, cv::INTER_LINEAR);

        dlib::matrix<dlib::rgb_pixel> face;
        dlib::assign_image(face, dlib::cv_image<dlib::bgr_pixel>(cvIplImage(resized_face)));
        // auto features = fr.computeRobustFaceDescriptor(face);
        auto features = face_recognition.computeFaceDescriptor(face);

        if (!index) {
          features_matrix.set_size(features.nr(), paths.size());
        }
        dlib::set_colm(features_matrix, index) = features;

        if (raw_embeddings.is_open()) {
          raw_embeddings << person << "," << path.lexically_relative(load_path);
          for (auto f : features) {
            raw_embeddings << "," << f;
          }
          raw_embeddings << std::endl;
        }

        ++index;
      }

      dlib::matrix<float> mean_features(features_matrix.nr(), 1);
      for (int row = 0; row < features_matrix.nr(); ++row) {
        mean_features(row) = dlib::mean(dlib::rowm(features_matrix, row));
      }
      std::cout << "  - Scene " << scene << " (mean): " << dlib::trans(mean_features);
      embeddings[id].emplace_back(mean_features);
    }
  }

  if (raw_embeddings.is_open()) {
    raw_embeddings.close();
  }

  std::cout << "Storing embeddings in " << output_path << std::endl;
  nlohmann::json j(embeddings);
  std::ofstream o(output_path);
  o << std::setw(4) << j << std::endl;

  std::cout << "The anonymous id mappings are:" << std::endl;
  for (const auto & [person, id] : person_to_id) {
    std::cout << person << ": " << id << std::endl;
  }

  return 0;
}
