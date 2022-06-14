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

