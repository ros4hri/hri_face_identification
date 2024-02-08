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

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include "fmt/core.h"
#include "hri/hri.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tclap/ArgException.h"
#include "tclap/CmdLine.h"
#include "tclap/ValueArg.h"

int main(int argc, char * argv[])
{
  std::string person_name;
  std::filesystem::path base_path;
  float rate;

  try {
    TCLAP::CmdLine cmd("Face samples recorder for pre-training face recognition", ' ');

    TCLAP::ValueArg<std::string> name_arg(
      "n", "name", "Name of the person to record", true, "", "string");
    cmd.add(name_arg);

    TCLAP::ValueArg<std::string> data_arg(
      "d", "data", "Path where the face images will be saved", false, "/tmp/face_dataset",
      "string");
    cmd.add(data_arg);

    TCLAP::ValueArg<float> rate_arg(
      "r", "rate", "Rate of capture (Hz)", false, 10., "float");
    cmd.add(rate_arg);

    cmd.parse(argc, argv);
    person_name = name_arg.getValue();
    base_path = data_arg.getValue();
    rate = rate_arg.getValue();

    if (person_name.empty()) {
      std::cerr << "Name cannot be empty" << std::endl;
      return 1;
    }
  } catch (TCLAP::ArgException & e) {
    std::cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
    return 1;
  }

  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("db_recorder");
  auto executor = rclcpp::executors::SingleThreadedExecutor::make_shared();
  executor->add_node(node);
  auto hri_listener = hri::HRIListener::create(node);

  cv::namedWindow("Face Detection", cv::WINDOW_FULLSCREEN & cv::WINDOW_KEEPRATIO);
  auto next_capture_time = std::chrono::steady_clock::now();
  int stream_nb = 0;
  int stream_frame_nb = 0;
  bool is_recording = false;
  bool new_stream = true;
  auto stream_path = base_path / person_name / std::to_string(stream_nb);
  int key = 0;

  while (true) {
    auto executor_thread = std::thread([&]() {executor->spin_all(std::chrono::milliseconds(100));});

    key = cv::waitKey(1);
    if (key == 32) {  // SPACE
      is_recording = !is_recording;
      new_stream = true;
    } else if (key == 27) {  // ESC
      break;
    }

    executor_thread.join();
    auto faces = hri_listener->getFaces();

    if (faces.size() > 1) {
      std::cout << "More then one face is detected! Skipping" << std::endl;
    } else if (faces.size() == 1) {
      auto id = faces.begin()->first;
      auto image = faces.begin()->second->aligned();

      if (image) {
        auto debug_image = image->clone();
        cv::putText(
          debug_image, fmt::format("{0}: frame {1:04d}", person_name, stream_frame_nb), {5, 10},
          cv::FONT_HERSHEY_SIMPLEX, 0.3, {255, 255, 255});

        if (is_recording) {
          if (new_stream) {
            while (
              std::filesystem::is_directory(stream_path) && std::filesystem::exists(stream_path))
            {
              ++stream_nb;
              stream_path = base_path / person_name / std::to_string(stream_nb);
            }
            std::filesystem::create_directories(stream_path);
            stream_frame_nb = 0;
            new_stream = false;
          }

          cv::putText(
            debug_image, "recording", {5, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.3, {255, 0, 0});
          cv::imwrite(stream_path / fmt::format("frame{:04d}.jpg", stream_frame_nb), *image);
          ++stream_frame_nb;
        }

        cv::imshow("Face Detection", debug_image);
      }
    }

    next_capture_time += std::chrono::milliseconds(static_cast<int>(1e3 / rate));
    std::this_thread::sleep_until(next_capture_time);
  }

  return 0;
}
