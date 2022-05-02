#include <dlib/dnn.h>
#include <dlib/pixel.h>

#include <map>
#include <opencv2/core.hpp>
#include <vector>

using Features = dlib::matrix<float, 0, 1>;

using Id = std::string;

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
    FaceRecognition(float match_threshold);

    void processFace(const cv::Mat& face);

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
     * The smaller the score, the better the match. As such, the best match
     * can be found with:
     *
     * ```cpp
     * auto scores = findCandidates(descriptor);
     * auto id = min_element(scores.begin(), scores.end(),
     *                      [](decltype(scores)::value_type& l,
     *                         decltype(scores)::value_type& r)
     *                         -> bool { return l.second < r.second;}
     *                      )->first;
     * ```
     */
    std::map<Id, float> findCandidates(Features descriptor);

    void storeFaceDB(std::string path) const;
    void loadFaceDB(std::string path);

   private:
    float computeConfidence(float distance) {
        return 1 - distance / match_threshold;
    }

    anet_type net;
    std::map<Id, std::vector<Features>> person_descriptors;
    std::map<Id, Id> face_person_map;

    float match_threshold;
};

