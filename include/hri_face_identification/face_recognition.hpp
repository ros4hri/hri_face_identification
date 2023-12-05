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

#include <diagnostic_updater/DiagnosticStatusWrapper.h>

#include <dlib/dnn.h>
#include <dlib/pixel.h>

#include <map>
#include <opencv2/core.hpp>
#include <vector>

using Features = dlib::matrix<float, 0, 1>;

using Id = std::string;

#define DEFAULT_MATCH_THRESHOLD 0.6

//////////////////////////////////////////////////////////////////////////////////
// construct the RESnet deep network with dlib
// copy-pasted from dlib public domain example dnn_face_recognition_ex.cpp

// note that the dlib model is trained for 150x150px images. As such, the below
// marcro *can not be changed*
#define IMG_SIZE 150

template <template <int, template <typename> class, int, typename> class block,
          int N, template <typename> class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block,
          int N, template <typename> class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<
    2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block =
    BN<dlib::con<N, 3, 3, 1, 1,
                 dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<
    128,
    dlib::avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<
        dlib::max_pool<3, 3, 2, 2,
                       dlib::relu<dlib::affine<dlib::con<
                           32, 7, 7, 2, 2,
                           dlib::input_rgb_image_sized<IMG_SIZE>>>>>>>>>>>>>;

//////////////////////////////////////////////////////////////////////////////////

class FaceRecognition {
   public:
    FaceRecognition(float match_threshold = DEFAULT_MATCH_THRESHOLD);

    /** tries to match a given face to the known faces.
     *
     * Returns a map {person_id, confidence} with all possible matches (ie,
     * face who are below the match threshold).
     *
     * Note that the match threshold is the maximum *distance* in the face
     * embedding space, while the returned *confidence* is a value between 1.0
     * (fully confident about the identification) and 0.0 (not confident at
     * all).
     *
     * If `create_person_if_needed` is true (default: false), a new person ID is
     * generated for the face, if the face does not match any known one.
     */
    std::map<Id, float> getAllMatches(const cv::Mat& face,
                                      bool create_person_if_needed = false);

    /** returns the best match between the provided face and the know faces,
     * with the associated confidence level (between 0.0 and 1.0).
     *
     * Effectively runs `processFace` and returns the candidate with the highest
     * confidence.
     *
     *
     * If `create_person_if_needed` is false (default),
     * returns an empty ID if not satisfactory match found.
     * Else, a new person ID will be generated and returned.
     */
    std::pair<Id, float> getBestMatch(const cv::Mat& face,
                                      bool create_person_if_needed = false);

    /** compute a face descriptor by projecting a face on the dlib's trained
     * facial recognition embedding.
     */
    Features computeFaceDescriptor(const dlib::matrix<dlib::rgb_pixel>& face);

    /** compute a more robust face descriptor by first generating jittered
     * versions of the provided face, and then computing the mean descriptor
     * on all resulting faces.
     *
     * Same API as computeFaceDescriptor, but accordingly much slower.
     */
    Features computeRobustFaceDescriptor(
        const dlib::matrix<dlib::rgb_pixel>& face);

    /** All this function does is make 100 copies of img, all slightly
     * jittered by being zoomed, rotated, and translated a little bit
     * differently. They are also randomly mirrored left to right.
     */
    std::vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(
        const dlib::matrix<dlib::rgb_pixel>& img);

    /** returns a map <person_id, score> for all the person whose face
     * descriptor(s) match (ie, distance < match_threshold) the provided
     * descriptor.
     *
     * The higher the score, the better the match. As such, the best match
     * can be found with:
     *
     * ```cpp
     * auto scores = findCandidates(descriptor);
     * auto id = max_element(scores.begin(), scores.end(),
     *                      [](decltype(scores)::value_type& l,
     *                         decltype(scores)::value_type& r)
     *                         -> bool { return l.second < r.second;}
     *                      )->first;
     * ```
     *
     */
    std::map<Id, float> findCandidates(Features descriptor);

    /** perform the module diagnostics filling a DiagnosticStatusWrapper 
     */
    void doDiagnostics(diagnostic_updater::DiagnosticStatusWrapper& status);

    /** stores the face database, in JSON format.
     */
    void storeFaceDB(std::string path) const;

    /** loads a face database in JSON format.
     */
    void loadFaceDB(std::string path);

    /** empties the face database. All previously identified persons will have
     * to be re-identified.
     *
     * Note that the on-disk database is not emptied/deleted.
     * Call FaceRecognition::storeFaceDB after calling
     * FaceRecognition::dropFaceDB to effectively delete all stored faces.
     */
    void dropFaceDB();

   private:
    float computeConfidence(float distance) {
        return 1 - distance / match_threshold;
    }

    anet_type net;
    std::map<Id, std::vector<Features>> person_descriptors;
    std::map<Id, Id> face_person_map;

    float match_threshold;
};

