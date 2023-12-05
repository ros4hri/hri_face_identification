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

#include "face_recognition.hpp"

#include <dlib/opencv.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <diagnostic_updater/DiagnosticStatusWrapper.h>
#include <time.h>

#include <random>
//#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>

#include "json.hpp"
using json = nlohmann::json;

using namespace std;

Id generate_id(const int len = 5) {
    // not a great implementation. Please suggest improvements if you feel like
    // it!
    static const array<string, 26> alphanum{
        {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
         "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}};
    string tmp_s;
    tmp_s.reserve(len);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<uint32_t> rnd_dist(0, alphanum.size() - 1);

    for (int i = 0; i < len; ++i) {
        tmp_s += alphanum[rnd_dist(rng)];
    }

    return tmp_s;
}

FaceRecognition::FaceRecognition(float _match_threshold)
    : match_threshold(_match_threshold) {
    ROS_INFO("Loading dlib's ANN face recognition resnet weights...");

    auto pkg_path_ = ros::package::getPath("hri_face_identification");

    // And finally we load the DNN responsible for face recognition.
    dlib::deserialize(pkg_path_ +
                      "/share/dlib_face_recognition_resnet_model_v1.dat") >>
        net;
}

std::map<Id, float> FaceRecognition::getAllMatches(
    const cv::Mat& cv_face, bool create_person_if_needed) {
    // wraps OpenCV image into a dlib image (no data copy)
    // ATTENTION: face->aligned() should not be modified while
    // wrapped: we first clone the image to ensure this does not
    // happen.
    // cv::Mat aligned_face(face->aligned().clone());
    cv::Mat resized_face;
    cv::resize(cv_face, resized_face, {IMG_SIZE, IMG_SIZE}, 0, 0,
               cv::INTER_LINEAR);
    // cv::imshow("input face", resized_face);
    // cv::waitKey(0);

    dlib::matrix<dlib::rgb_pixel> face;
    dlib::assign_image(
        face, dlib::cv_image<dlib::bgr_pixel>(cvIplImage(resized_face)));

    // calculate the face's descriptor by projecting the image on the RESnet
    // embedding
    auto desc = computeFaceDescriptor(face);

    // search for known faces that are close to the new face in the embedding
    // space
    auto candidates = findCandidates(desc);

    if (candidates.empty()) {
        if (create_person_if_needed) {
            Id person_id = generate_id();

            ROS_INFO_STREAM("New person detected; will be identified under id <"
                            << person_id << ">");

            ROS_INFO_STREAM("Computing face descriptor...");
            person_descriptors[person_id].push_back(
                // computeRobustFaceDescriptor(face));
                computeFaceDescriptor(face));
            ROS_INFO_STREAM("done!");

            return {{person_id, 1.0}};
        } else
            return {};
    } else {
        if (candidates.size() == 1) {
            auto& kv = *candidates.begin();
            // we've got a match!
            ROS_INFO_STREAM("Found a match with person "
                            << kv.first << " (confidence: " << kv.second);

            // compute & add new face descriptor for that person if too
            // far from the original one, but not too bad either (to avoid
            // adding too many false positive)
            if (kv.second < 0.6 && kv.second > 0.4) {
                ROS_INFO_STREAM("Current score: "
                                << kv.second
                                << ". Adding an additional face descriptor for "
                                << kv.first);
                person_descriptors[kv.first].push_back(
                    computeFaceDescriptor(face));
            }

        } else {
            ROS_INFO("Found more than one possible match:");

            for (const auto& kv : candidates) {
                ROS_INFO_STREAM("  - " << kv.first << " (c=" << kv.second
                                       << ")");
            }
        }

        return candidates;
    }
}

pair<Id, float> FaceRecognition::getBestMatch(const cv::Mat& face,
                                              bool create_person_if_needed) {
    auto candidates = getAllMatches(face, create_person_if_needed);

    if (candidates.empty()) {
        return make_pair(Id(), 0.0);
    }

    auto best = max_element(candidates.begin(), candidates.end(),
                            [](decltype(candidates)::value_type& l,
                               decltype(candidates)::value_type& r) -> bool {
                                return l.second < r.second;
                            });

    return make_pair(best->first, best->second);
}

Features FaceRecognition::computeFaceDescriptor(
    const dlib::matrix<dlib::rgb_pixel>& face) {
    //////////////////////////////////////////
    // dlib::matrix<dlib::rgb_pixel> face_tmp;
    // dlib::assign_image(face_tmp, face);
    // cv::imshow("face", dlib::toMat(face_tmp));
    // cv::waitKey(0);
    /////////////////////////////////////////

    return net(face);
}

Features FaceRecognition::computeRobustFaceDescriptor(
    const dlib::matrix<dlib::rgb_pixel>& face) {
    dlib::matrix<float, 0, 1> face_descriptor =
        mean(mat(net(jitter_image(face))));

    ROS_DEBUG_STREAM(
        "jittered face descriptor for one face: " << trans(face_descriptor));

    return face_descriptor;
}

vector<dlib::matrix<dlib::rgb_pixel>> FaceRecognition::jitter_image(
    const dlib::matrix<dlib::rgb_pixel>& img) {
    thread_local dlib::rand rnd;

    vector<dlib::matrix<dlib::rgb_pixel>> crops;
    for (int i = 0; i < 100; ++i) crops.push_back(dlib::jitter_image(img, rnd));

    return crops;
}

map<Id, float> FaceRecognition::findCandidates(Features descriptor) {
    map<Id, float> scores;

    for (const auto& kv : person_descriptors) {
        auto person_id = kv.first;
        for (const auto& known_descriptor : kv.second) {
            auto distance = length(descriptor - known_descriptor);

            if (distance < match_threshold) {
                auto score = computeConfidence(distance);

                if (scores.count(person_id) == 0 or scores[person_id] < score) {
                    // first match or new best match
                    scores[person_id] = score;
                }
            }
        }
    }

    return scores;
}

// custon JSON serializers for dlib's vectors
namespace dlib {
void to_json(json& j, const Features& f) {
    std::vector<float> features;

    for (unsigned int r = 0; r < f.nr(); r += 1) {
        features.push_back(f(r, 0));
    }

    j = json(features);
}

void from_json(const json& j, Features& f) {
    std::vector<float> features;
    j.get_to(features);

    f = dlib::mat(features);
}
}  // namespace dlib

void FaceRecognition::doDiagnostics(diagnostic_updater::DiagnosticStatusWrapper& status) {
    status.summary(diagnostic_msgs::DiagnosticStatus::OK, "");
    status.add("Known faces", person_descriptors.size());
    status.add("Last recognized face ID", person_descriptors.rbegin()->first);
}

void FaceRecognition::storeFaceDB(string path) const {
    json j(person_descriptors);

    std::ofstream o(path);
    o << std::setw(4) << j << std::endl;

    cout << "Face database correctly saved to " << path << endl;
}

void FaceRecognition::loadFaceDB(string path) {
    std::ifstream i(path);
    if (i.good()) {
        json j;
        i >> j;

        auto new_descriptors = j.get<map<Id, std::vector<Features>>>();
        person_descriptors.insert(new_descriptors.begin(),
                                  new_descriptors.end());

        ROS_INFO_STREAM("Face database correctly loaded from " << path);
    } else {
        ROS_WARN_STREAM("Unable to load face database from "
                        << path << ". Starting with no known faces.");
    }
}

void FaceRecognition::dropFaceDB() { person_descriptors.clear(); }

