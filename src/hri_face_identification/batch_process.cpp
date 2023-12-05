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

#include <dlib/opencv.h>
#include <ros/console.h>

#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "face_recognition.hpp"

using namespace std;
using namespace filesystem;

const vector<string> EXTENSIONS({".jpg", ".jpeg", ".png", ".JPG", ".JPEG"});

int main(int argc, char** argv) {
    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                       ros::console::levels::Warn)) {
        ros::console::notifyLoggerLevelsChanged();
    }

    if (argc < 2) {
        cerr << "you need to pass the path the face dataset (directory with "
                "one subdirectory per person, each containing cropped images "
                "of the person's face)"
             << endl;
        return 1;
    }

    path base_path(argv[1]);

    cerr << "Processing images in " << base_path << endl;

    float match_threshold = 0.6;

    FaceRecognition fr(match_threshold);

    string current_person;
    path current_path;

    for (const auto& dirEntry : recursive_directory_iterator(base_path)) {
        if (dirEntry.is_directory()) {
            if (dirEntry.path().parent_path() == base_path) {
                current_path = dirEntry.path();
                current_person = current_path.filename();
                cerr << "Processing " << current_person << "..." << endl;
            }
        } else {
            bool is_image =
                (std::find(EXTENSIONS.begin(), EXTENSIONS.end(),
                           dirEntry.path().extension()) != EXTENSIONS.end());
            if (is_image) {
                cerr << "  - Frame "
                     << dirEntry.path().lexically_relative(current_path)
                     << endl;

                cv::Mat cv_face = cv::imread(dirEntry.path());
                cv::Mat resized_face;
                cv::resize(cv_face, resized_face, {IMG_SIZE, IMG_SIZE}, 0, 0,
                           cv::INTER_LINEAR);

                dlib::matrix<dlib::rgb_pixel> face;
                dlib::assign_image(face, dlib::cv_image<dlib::bgr_pixel>(
                                             cvIplImage(resized_face)));
                // auto features = fr.computeRobustFaceDescriptor(face);
                auto features = fr.computeFaceDescriptor(face);

                cout << current_person << ","
                     << dirEntry.path().lexically_relative(current_path);
                for (auto f : features) {
                    cout << "," << f;
                }
                cout << endl;
            } else {
                cerr << "Skipping " << dirEntry << ": not an image extension"
                     << endl;
            }
        }
    }
    return 0;
}

