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

using namespace std;

class ROS4HRIFaceIdentificationTest : public ::testing::Test {
   public:
    ROS4HRIFaceIdentificationTest() {
        pkg_path_ = ros::package::getPath("hri_face_identification");
        test_dir_ = pkg_path_ + "/test-data/";
    }
    ~ROS4HRIFaceIdentificationTest() {}

    // void SetUp() override{}
    // void TearDown() override{}

    std::string pkg_path_;
    std::string test_dir_;

    FaceRecognition fr;

   private:
    ros::NodeHandle nh_;
};

TEST_F(ROS4HRIFaceIdentificationTest, SinglePerson) {
    rosbag::Bag bag;

    string face_id = "2374d";
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
            auto res = fr.processFace(cv_face);
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
//
// TEST_F(ROS4HRIFaceIdentificationTest, DISABLED_ConsistentFacesBag) {
//    ros::NodeHandle nh;
//
//    std::vector<std::string> bags = {"severin_face_gt.bag"};
//
//    for (const std::string &bag_file : bags) {
//        std::string bag_name = "../bags/" + bag_file;
//
//        SCOPED_TRACE(bag_name);
//
//        // playRosbag(test_dir_ + bag_name);
//
//        rosbag::Bag bag;
//
//        bag.open(test_dir_ + bag_name, rosbag::bagmode::Read);
//        rosbag::View view(bag, rosbag::TopicQuery(FACES));
//
//        size_t total_faces = 0;
//        size_t total_rois = 0;
//        for (const auto &m : view) {
//            auto msg = m.instantiate<hri_msgs::IdsList>();
//            total_faces += msg->ids.size();
//        }
//
//        rosbag::View face_view(
//            bag, ROS4HRITopicsQuery(ROS4HRITopicsQuery::Subtopic::face,
//            "roi"));
//
//        for (const auto &f : face_view) {
//            auto roi = f.instantiate<sensor_msgs::RegionOfInterest>();
//            total_rois += 1;
//            EXPECT_TRUE(roi->width > 0);
//            EXPECT_TRUE(roi->height > 0);
//        }
//
//        // check the 'faces/tracked' list of face IDs matches the
//        // number of faces actually published
//        EXPECT_EQ(total_faces, total_rois);
//    }
//}
//
// TEST_F(ROS4HRIFaceIdentificationTest, DISABLED_FaceIdentification) {
//    ros::NodeHandle nh;
//
//    std::vector<std::string> bags = {"severin_face_gt.bag"};
//
//    for (const std::string &bag_file : bags) {
//        std::string bag_name = "../bags/" + bag_file;
//
//        ROS_INFO_STREAM("[II] Processing " << bag_name);
//        SCOPED_TRACE(bag_name);
//
//        // playRosbag(test_dir_ + bag_name);
//
//        rosbag::Bag bag;
//
//        bag.open(test_dir_ + bag_name, rosbag::bagmode::Read);
//        rosbag::View view(bag, rosbag::TopicQuery(RGB_COMPRESSED));
//
//        size_t nb_imgs = 0;
//        size_t missed_frames = 0;
//
//        for (const auto &m : view) {
//            if (m.getTopic() == RGB_COMPRESSED) {
//                auto current_ = m.getTime();
//
//                auto compressed_rgb =
//                    m.instantiate<sensor_msgs::CompressedImage>();
//
//                if (compressed_rgb != NULL) {
//                    nb_imgs += 1;
//
//                    // decompress image & publish it
//                    auto cvimg = cv::imdecode(compressed_rgb->data, 1);
//                    auto img_msg =
//                    cv_bridge::CvImage(compressed_rgb->header,
//                                                      "bgr8", cvimg)
//                                       .toImageMsg();
//                    img_publisher_.publish(img_msg);
//
//                    // res contains the list of detected face ids
//                    // (hri_msgs::IdsList)
//                    auto res = waitForFaces();
//
//                    if (!res) {
//                        missed_frames += 1;
//                        continue;
//                    }
//
//                    // faces msg MUST reuse the timestamp of the original
//                    image ASSERT_EQ(res->header.stamp,
//                    img_msg->header.stamp);
//
//                    rosbag::View face_view(bag, rosbag::TopicQuery(FACES),
//                                           current_ - ros::Duration(1, 0),
//                                           current_ + ros::Duration(1,
//                                           0));
//
//                    for (rosbag::MessageInstance const f : face_view) {
//                        auto groundtruth =
//                        f.instantiate<hri_msgs::IdsList>(); if
//                        (groundtruth->header.stamp ==
//                            img_msg->header.stamp) {
//                            EXPECT_EQ(res->ids.size(),
//                            groundtruth->ids.size())
//                                << "on frame " << nb_imgs << " (" <<
//                                current_
//                                << "), # of detected face do not match
//                                ground
//                                "
//                                   "truth";
//                        }
//                    }
//                }
//            }
//        }
//
//        ROS_INFO_STREAM("[II] Processed " << nb_imgs << " frames");
//
//        // first frame is always missed -> connection delay between
//        publishers
//        // and subscribers?
//        ASSERT_LE(missed_frames, 1) << "Frames where missed";
//    }
//}
//
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "ros4hri_face_identification_test");
    ros::NodeHandle nh;
    return RUN_ALL_TESTS();
}

