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

#include <chrono>
#include <cstdio>

#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "opencv2/imgcodecs.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "hri_face_identification/face_recognition.hpp"

class FaceRecognitionTest : public ::testing::Test
{
protected:
  FaceRecognitionTest()
  {
    std::filesystem::path share_path(
      ament_index_cpp::get_package_share_directory("hri_face_identification"));
    face_db_path_ = share_path / "model" / "data" / "faces_db.json";

    std::ifstream face_db_stream(face_db_path_.string());
    if (face_db_stream.good()) {
      nlohmann::json face_db_json;
      face_db_stream >> face_db_json;
      for (auto const & [person, descriptors] : face_db_json.items()) {
        // we know the id of the default database, we use also as person name
        default_id_to_person_.emplace(person, person);
      }
    }

    face_recognition_ = std::make_unique<hri_face_identification::FaceRecognition>(
      (share_path / "model" / "dlib_face_recognition_resnet_model_v1.dat").string(), 0.5);
  }

  void SetUp() override
  {
    face_recognition_->loadFaceDB(face_db_path_.string());
    id_to_person_ = default_id_to_person_;
  }

  void TearDown() override
  {
    face_recognition_->dropFaceDB();
    id_to_person_.clear();
  }

  void recognize(const std::vector<std::filesystem::path> & relative_images_paths)
  {
    std::vector<std::set<std::filesystem::path>> images_paths_sets;

    for (const auto & path : relative_images_paths) {
      images_paths_sets.emplace_back();
      auto abs_path = std::filesystem::current_path() / path;
      for (auto const & dir_entry : std::filesystem::recursive_directory_iterator(abs_path)) {
        if (dir_entry.is_regular_file()) {
          images_paths_sets.back().insert(dir_entry.path());
        }
      }
    }

    for (auto const & images_paths_set : images_paths_sets) {
      for (auto const & img_path : images_paths_set) {
        std::string person{};
        for (auto it = img_path.begin(); it != img_path.end(); ++it) {
          if (it->string().find("person_") != std::string::npos) {
            person = it->string();
          }
        }
        ASSERT_FALSE(person.empty()) << "Failed to find the person id 'person_*' in path " <<
          img_path;

        auto cv_face = cv::imread(img_path.string());
        bool new_person_created;
        bool should_create_person{true};
        for (auto const & [id, mapped_person] : id_to_person_) {
          if (person == mapped_person) {
            should_create_person = false;
            break;
          }
        }
        auto [id, confidence] = face_recognition_->getBestMatch(
          cv_face, should_create_person, new_person_created);

        if (should_create_person) {
          if (new_person_created) {
            ASSERT_FALSE(id.empty()) << "For image " << img_path <<
              " a new person is created but no id is returned";
            id_to_person_[id] = person;
          }
        } else {
          ASSERT_FALSE(new_person_created) << "A non-allowed new person is created";
        }

        EXPECT_FALSE(id.empty()) << "Failed to recognize the image " << img_path;
        if (!id.empty()) {
          EXPECT_EQ(person, id_to_person_[id]) << "The image " << img_path <<
            " is confused with " << id_to_person_[id] << " with confidence " << confidence;
        }
      }
    }
  }

  std::unique_ptr<hri_face_identification::FaceRecognition> face_recognition_;

private:
  std::map<std::string, hri_face_identification::Id> id_to_person_;
  std::map<std::string, hri_face_identification::Id> default_id_to_person_;
  std::filesystem::path face_db_path_;
};

TEST_F(FaceRecognitionTest, SinglePerson)
{
  recognize({"data/person_0/scene_0"});
}

TEST_F(FaceRecognitionTest, SinglePersonMultipleFiles)
{
  recognize({"data/person_0"});
}

TEST_F(FaceRecognitionTest, MultiPerson)
{
  recognize({"data", "data"});  // test all images twice in two separate cycles
}

TEST_F(FaceRecognitionTest, DBStoreLoad)
{
  recognize({"data/person_0/scene_0"});
  auto temp_db_path = (std::filesystem::temp_directory_path() / "test_faces_db.json").string();
  face_recognition_->storeFaceDB(temp_db_path);
  face_recognition_->dropFaceDB();
  face_recognition_->loadFaceDB(temp_db_path);
  recognize({"data/person_0/scene_0"});
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
