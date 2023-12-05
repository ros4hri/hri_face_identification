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

#include <cv_bridge/cv_bridge.h>
#include <gtest/gtest.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>

#include <string>
#include <vector>

#include "../src/face_recognition.hpp"
#include "../src/json.hpp"
using json = nlohmann::json;

using namespace std;

const float CONFIDENCE_THRESHOLD = 0.3;

class ROS4HRIFaceIdentificationTest : public ::testing::Test {
   public:
    ROS4HRIFaceIdentificationTest() {
        pkg_path_ = ros::package::getPath("hri_face_identification");
        test_dir_ = pkg_path_ + "/test-data/";

        std::ifstream input(test_dir_ + "/face_person_map.json");

        if (input.good()) {
            json j;
            input >> j;

            j.get_to(face_person_groups);

            int i = 0;
            for (auto& group : face_person_groups) {
                for (auto& face_id : group) {
                    face_person_map[face_id] = i;
                }
                i++;
            }
        }
    }

    ~ROS4HRIFaceIdentificationTest() {}

    void SetUp() override { fr.dropFaceDB(); }
    // void TearDown() override{}

    std::string pkg_path_;
    std::string test_dir_;

    FaceRecognition fr;

    std::map<std::string, int> face_person_map;
    std::vector<std::vector<std::string>> face_person_groups;

   private:
    ros::NodeHandle nh_;
};

TEST_F(ROS4HRIFaceIdentificationTest, SinglePerson) {
    rosbag::Bag bag;

    string face_id = "932a0";
    bag.open(test_dir_ + string("face_") + face_id + ".bag",
             rosbag::bagmode::Read);

    std::vector<std::string> topics;
    topics.push_back(std::string("/humans/faces/" + face_id + "/aligned"));

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    string person_id;

    size_t idx = 0;

    for (rosbag::MessageInstance const m : view) {
        sensor_msgs::Image::ConstPtr face = m.instantiate<sensor_msgs::Image>();
        if (face != NULL) {
            ROS_INFO_STREAM("Testing frame " << idx);
            idx++;

            auto cv_face = cv_bridge::toCvCopy(face)->image;
            auto res = fr.getAllMatches(cv_face, true);
            EXPECT_EQ(res.size(), 1);

            if (person_id.empty()) {
                person_id = (*res.begin()).first;
            } else {
                EXPECT_EQ((*res.begin()).first, person_id);
            }
        }
    }
    bag.close();
}

TEST_F(ROS4HRIFaceIdentificationTest, SinglePersonMultipleFiles) {
    rosbag::Bag bag;

    size_t face_idx = 0;
    std::map<int, std::string> face_person_result_map;
    auto group =
        face_person_groups[0];  // 5 different bag files, recorded under
                                // different conditions, of the same person

    int total_faces = 0;
    int unsure_matches = 0;

    for (auto& face_id : group) {
        ROS_INFO_STREAM("Processing bag " << test_dir_ << "/face_" << face_id
                                          << ".bag");

        int expected_id = face_person_map[face_id];
        ROS_INFO_STREAM("This bag should contain face " << expected_id);

        bag.open(test_dir_ + string("face_") + face_id + ".bag",
                 rosbag::bagmode::Read);

        std::vector<std::string> topics;
        topics.push_back(std::string("/humans/faces/" + face_id + "/aligned"));

        rosbag::View view(bag, rosbag::TopicQuery(topics));

        size_t idx = 0;

        std::vector<std::string> known_person_ids;

        for (rosbag::MessageInstance const m : view) {
            sensor_msgs::Image::ConstPtr face =
                m.instantiate<sensor_msgs::Image>();
            if (face != NULL) {
                ROS_INFO_STREAM("Testing frame " << idx);

                total_faces += 1;

                auto cv_face = cv_bridge::toCvCopy(face)->image;
                auto best = fr.getBestMatch(
                    cv_face,
                    true);  // true means that a new person_id will be
                            // created if the face is not identified

                auto person_id = best.first;
                auto confidence = best.second;

                ASSERT_NE(person_id.size(),
                          0);  // make sure an ID is returned

                if (face_person_result_map.count(expected_id) == 0) {
                    // it should be a new person!
                    EXPECT_TRUE(std::find(known_person_ids.begin(),
                                          known_person_ids.end(),
                                          person_id) == known_person_ids.end());
                    face_person_result_map[expected_id] = person_id;
                    known_person_ids.push_back(person_id);
                } else {
                    if (confidence < CONFIDENCE_THRESHOLD) {
                        ROS_INFO_STREAM("unsure match on frame " << idx);
                        unsure_matches += 1;
                    } else {
                        EXPECT_EQ(person_id,
                                  face_person_result_map[expected_id]);
                    }
                }

                idx++;
            }
        }
        bag.close();
    }
    ROS_WARN_STREAM(
        "[Single person, multi-bags] Total number of 'unsure' matches:"
        << unsure_matches << " (" << (100. * unsure_matches / total_faces)
        << "% of " << total_faces << " total faces)");
}

TEST_F(ROS4HRIFaceIdentificationTest, MultiPerson) {
    rosbag::Bag bag;

    size_t face_idx = 0;
    std::map<int, std::string> face_person_result_map;

    const size_t SKIP_FRAMES = 10;
    size_t skipped_frames = 0;

    int total_faces = 0;
    int unsure_matches = 0;

    const size_t NB_FOLDS = 2;

    for (int fold = 1; fold <= NB_FOLDS; fold++) {
        ROS_INFO_STREAM("Running fold " << fold << " of " << NB_FOLDS);

        for (auto& group : face_person_groups) {
            for (auto& face_id : group) {
                ROS_INFO_STREAM("Processing bag " << test_dir_ << "/face_"
                                                  << face_id << ".bag");

                int expected_id = face_person_map[face_id];
                ROS_INFO_STREAM("This bag should contain face " << expected_id);

                bag.open(test_dir_ + string("face_") + face_id + ".bag",
                         rosbag::bagmode::Read);

                std::vector<std::string> topics;
                topics.push_back(
                    std::string("/humans/faces/" + face_id + "/aligned"));

                rosbag::View view(bag, rosbag::TopicQuery(topics));

                size_t idx = 0;

                std::vector<std::string> known_person_ids;

                for (rosbag::MessageInstance const m : view) {
                    skipped_frames += 1;
                    if ((skipped_frames % SKIP_FRAMES) != 0) {
                        continue;
                    } else {
                        skipped_frames = 0;
                    }

                    sensor_msgs::Image::ConstPtr face =
                        m.instantiate<sensor_msgs::Image>();
                    if (face != NULL) {
                        ROS_INFO_STREAM("Testing frame " << idx);

                        total_faces += 1;

                        auto cv_face = cv_bridge::toCvCopy(face)->image;
                        auto best = fr.getBestMatch(
                            cv_face,
                            true);  // true means that a new person_id will be
                                    // created if the face is not identified

                        auto person_id = best.first;
                        auto confidence = best.second;

                        ASSERT_NE(person_id.size(),
                                  0);  // make sure an ID is returned

                        if (face_person_result_map.count(expected_id) == 0) {
                            // it should be a new person!
                            EXPECT_TRUE(std::find(known_person_ids.begin(),
                                                  known_person_ids.end(),
                                                  person_id) ==
                                        known_person_ids.end());
                            face_person_result_map[expected_id] = person_id;
                            known_person_ids.push_back(person_id);
                        } else {
                            if (confidence < CONFIDENCE_THRESHOLD) {
                                ROS_INFO_STREAM("unsure match on frame "
                                                << idx);
                                unsure_matches += 1;
                            } else {
                                EXPECT_EQ(person_id,
                                          face_person_result_map[expected_id]);
                            }
                        }

                        idx++;
                    }
                }
                bag.close();
            }
        }
    }

    ROS_WARN_STREAM("[Multiple people] Total number of 'unsure' matches:"
                    << unsure_matches << " ("
                    << (100. * unsure_matches / total_faces) << "% of "
                    << total_faces << " total faces)");
}

TEST_F(ROS4HRIFaceIdentificationTest, DBStoreLoad) {
    rosbag::Bag bag;

    string face_id = "932a0";
    bag.open(test_dir_ + string("face_") + face_id + ".bag",
             rosbag::bagmode::Read);

    std::vector<std::string> topics;
    topics.push_back(std::string("/humans/faces/" + face_id + "/aligned"));

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    string person_id;

    size_t idx = 0;

    sensor_msgs::Image::ConstPtr face;

    face = (*view.begin()).instantiate<sensor_msgs::Image>();

    ASSERT_TRUE(face != NULL);

    auto cv_face = cv_bridge::toCvCopy(face)->image;
    auto res = fr.getAllMatches(cv_face, true);

    ASSERT_EQ((*res.begin()).second, 1.0)
        << "Should be a newly detected face, eg confidence = 1";
    person_id = (*res.begin()).first;

    fr.storeFaceDB("/tmp/hri_face_identification_test.json");

    fr.dropFaceDB();

    // no faces anymore: we should re-generate a new random face id
    res = fr.getAllMatches(cv_face, true);

    ASSERT_EQ((*res.begin()).second, 1.0)
        << "Should be again a newly detected face, eg confidence = 1";

    ASSERT_NE((*res.begin()).first, person_id)
        << "A new random id should have been generated";

    fr.dropFaceDB();
    fr.loadFaceDB("/tmp/hri_face_identification_test.json");

    // this time, we've re-loaded the face db: we should re-detect the original
    // person
    res = fr.getAllMatches(cv_face, true);

    ASSERT_EQ((*res.begin()).first, person_id);

    bag.close();
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "ros4hri_face_identification_test");
    ros::NodeHandle nh;
    return RUN_ALL_TESTS();
}

