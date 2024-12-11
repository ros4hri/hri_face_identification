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

#include "hri_face_identification/face_recognition.hpp"

#include <time.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <utility>

#include "dlib/dnn.h"
#include "dlib/matrix.h"
#include "dlib/rand.h"
#include "dlib/opencv.h"
#include "nlohmann/json.hpp"
#include "opencv2/imgproc.hpp"

namespace hri_face_identification
{

Id generateId(const int len = 5)
{
  // not a great implementation. Please suggest improvements if you feel like it!
  static const std::array<std::string, 26> alphanum{
    {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
      "t", "u", "v", "w", "x", "y", "z"}};
  std::string tmp_s;
  tmp_s.reserve(len);

  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<uint32_t> rnd_dist(0, alphanum.size() - 1);

  for (int i = 0; i < len; ++i) {
    tmp_s += alphanum[rnd_dist(rng)];
  }

  return tmp_s;
}

FaceRecognition::FaceRecognition(
  std::string model_path,
  float distance_threshold)
: distance_threshold_(distance_threshold)
{
  dlib::deserialize(model_path) >> net_;
}

std::vector<std::pair<Id, float>> FaceRecognition::getAllMatches(
  const cv::Mat & cv_face,
  bool create_person_if_needed,
  bool & new_person_created)
{
  new_person_created = false;

  // wraps OpenCV image into a dlib image (no data copy)
  // ATTENTION: cv_face should not be modified while wrapped.
  // we first clone the image to ensure this does not happen.
  // cv::Mat cv_face(original_cv_face.clone());
  // auto results = face_recognition.getAllMatches(cv_face, create_person_if_needed);
  cv::Mat resized_face;
  cv::resize(cv_face, resized_face, {IMG_SIZE, IMG_SIZE}, 0, 0, cv::INTER_LINEAR);

  DLibImage face;
  dlib::assign_image(face, dlib::cv_image<dlib::bgr_pixel>(cvIplImage(resized_face)));

  // calculate the face's descriptor by projecting the image on the RESnet embedding
  auto desc = computeFaceDescriptor(face);

  // search for known faces that are close to the new face in the embedding space
  auto candidates = findCandidates(desc);

  if (candidates.empty()) {
    if (create_person_if_needed) {
      // new person detected, we add it to the database with a new ID
      Id person_id = generateId();
      person_descriptors_[person_id].push_back(computeFaceDescriptor(face));
      new_person_created = true;
      return {{person_id, 1.0}};
    } else {
      return {};
    }
  } else {
    if (candidates.size() == 1) {
      auto &[id, confidence] = *candidates.begin();
      // we've got a match!
      // compute & add new face descriptor for that person if too far from the
      // original one, but not too bad either (to avoid adding too many false
      // positive)
      if (confidence < 0.6 && confidence > 0.4) {
        person_descriptors_[id].push_back(computeFaceDescriptor(face));
        pruneDescriptors(person_descriptors_[id]);
      }
    }

    std::map<float, Id> ascending_candidates;
    for (const auto & [id, confidence] : candidates) {
      ascending_candidates[confidence] = id;
    }
    std::vector<std::pair<Id, float>> results;
    for (auto it = ascending_candidates.rbegin(); it != ascending_candidates.rend(); ++it) {
      results.emplace_back(it->second, it->first);
    }

    return results;
  }
}

Features FaceRecognition::computeFaceDescriptor(const DLibImage & face)
{
  return net_(face);
}

Features FaceRecognition::computeRobustFaceDescriptor(const DLibImage & face)
{
  return dlib::mean(dlib::mat(net_(jitterImage(face))));
}

std::vector<DLibImage> FaceRecognition::jitterImage(const DLibImage & img)
{
  thread_local dlib::rand rnd;

  std::vector<DLibImage> crops;
  for (int i = 0; i < 100; ++i) {
    crops.push_back(dlib::jitter_image(img, rnd));
  }

  return crops;
}

/** compute the centroid of all the descriptors of a person,
 * then iterate over all the descriptors and remove the one
 * that is the farthest from the centroid.
 */
void FaceRecognition::pruneDescriptors(std::vector<Features> descriptors)
{
  if (descriptors.size() < MAX_FACE_DESCRIPTORS) {
    return;
  }

  // compute the centroid
  Features centroid = dlib::mean(dlib::mat(descriptors));

  // find the descriptor that is the farthest from the centroid
  auto worst_descriptor = std::max_element(
    descriptors.begin(), descriptors.end(),
    [&centroid](const Features & l, const Features & r) -> bool {
      return dlib::length(l - centroid) < dlib::length(r - centroid);
    });

  // remove it
  descriptors.erase(worst_descriptor);
}

std::map<Id, float> FaceRecognition::findCandidates(Features descriptor)
{
  std::map<Id, float> scores;

  for (const auto &[person_id, known_descriptors] : person_descriptors_) {
    for (const auto & known_descriptor : known_descriptors) {
      auto distance = dlib::length(descriptor - known_descriptor);

      if (distance < distance_threshold_) {
        auto score = computeConfidence(distance);

        if (scores.count(person_id) == 0 || scores[person_id] < score) {
          // first match or new best match
          scores[person_id] = score;
        }
      }
    }
  }

  return scores;
}

/** Converts a distance in embedding space to a confidence level between 0 and 1.
 * The confidence level is computed as a polynomial of degree 4, with the
 * following properties:
 * - confidence = 1 at distance = 0
 * - confidence = 0 at distance >= match_distance_threshold
 *   (ie, the confidence is 0 when the distance is equal to the threshold)
 */
float FaceRecognition::computeConfidence(float distance)
{
  if (distance > distance_threshold_) {
    return 0;
  }

  return 1.f - pow(distance / distance_threshold_, 4);
}

FaceRecognitionDiagnostics FaceRecognition::getDiagnostics()
{
  FaceRecognitionDiagnostics diagnostics{};
  if (!person_descriptors_.empty()) {
    diagnostics.known_persons = static_cast<int>(person_descriptors_.size());
    diagnostics.last_person_id = person_descriptors_.rbegin()->first;
  }
  return diagnostics;
}

void FaceRecognition::storeFaceDB(std::string path) const
{
  nlohmann::json j(person_descriptors_);

  std::ofstream o(path);
  o << std::setw(4) << j << std::endl;
}

bool FaceRecognition::loadFaceDB(std::string path)
{
  std::ifstream i(path);
  if (i.good()) {
    nlohmann::json j;
    i >> j;

    auto new_descriptors = j.get<std::map<Id, std::vector<Features>>>();
    person_descriptors_.insert(new_descriptors.begin(), new_descriptors.end());
    return true;
  } else {
    return false;
  }
}

void FaceRecognition::dropFaceDB() {person_descriptors_.clear();}

}  // namespace hri_face_identification

// custom JSON serializers for dlib's vectors
namespace dlib
{
// using json = nlohmann::json;

void to_json(nlohmann::json & j, const hri_face_identification::Features & f)
{
  std::vector<float> features;

  for (unsigned int r = 0; r < f.nr(); r += 1) {
    features.push_back(f(r, 0));
  }

  j = nlohmann::json(features);
}

void from_json(const nlohmann::json & j, hri_face_identification::Features & f)
{
  std::vector<float> features;
  j.get_to(features);

  f = dlib::mat(features);
}
}  // namespace dlib
