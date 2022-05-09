// Copyright 2022 PAL Robotics S.L.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the PAL Robotics S.L. nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

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

        std::ifstream i(test_dir_ + "/face_person_map.json");

        if (i.good()) {
            json j;
            i >> j;

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
            auto res = fr.processFace(cv_face, true);
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
                idx++;

                total_faces += 1;

                auto cv_face = cv_bridge::toCvCopy(face)->image;
                auto best = fr.bestMatch(
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
                        unsure_matches += 1;
                    } else {
                        EXPECT_EQ(person_id,
                                  face_person_result_map[expected_id]);
                    }
                }
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
                        idx++;

                        total_faces += 1;

                        auto cv_face = cv_bridge::toCvCopy(face)->image;
                        auto best = fr.bestMatch(
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
                                unsure_matches += 1;
                            } else {
                                EXPECT_EQ(person_id,
                                          face_person_result_map[expected_id]);
                            }
                        }
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

}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "ros4hri_face_identification_test");
    ros::NodeHandle nh;
    return RUN_ALL_TESTS();
}

