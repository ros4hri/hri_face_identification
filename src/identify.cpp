#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <dlib/pixel.h>
#include <hri/face.h>
#include <hri/hri.h>
#include <hri_msgs/IdsMatch.h>
#include <ros/ros.h>

#include <cstdlib>  // for rand()
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace ros;
using namespace hri;
using namespace std;

// if the distance between 2 facial descriptors is less than this threshold, we
// consider that the 2 faces are indeed the same person.
const float EUCLIDIAN_THRESHOLD = 0.6;

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

using Features = dlib::matrix<float, 0, 1>;

using Id = std::string;

//////////////////////////////////////////////////////////////////////////////////

Id generate_id(const int len = 5) {
    static const char alphanum[] = "0123456789abcdef";
    string tmp_s;
    tmp_s.reserve(len);

    for (int i = 0; i < len; ++i) {
        tmp_s += alphanum[::rand() % (sizeof(alphanum) - 1)];
    }

    return tmp_s;
}

// ----------------------------------------------------------------------------------------

class FaceRecogntion {
   public:
    FaceRecogntion() {
        ROS_INFO("Loading dlib's ANN face recognition resnet weights...");
        // And finally we load the DNN responsible for face recognition.
        dlib::deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
    }

    Features computeFaceDescriptor(dlib::matrix<dlib::rgb_pixel> face) {
        return net(face);
    }

    /** compute a more robust face descriptor by first generating jittered
     * versions of the provided face, and then computing the mean descriptor on
     * all resulting faces.
     *
     * Same API as computeFaceDescriptor, but accordingly much slower.
     */
    Features computeRobustFaceDescriptor(dlib::matrix<dlib::rgb_pixel> face) {
        dlib::matrix<float, 0, 1> face_descriptor =
            mean(mat(net(jitter_image(face))));
        cout << "jittered face descriptor for one face: "
             << trans(face_descriptor) << endl;

        return face_descriptor;
    }

    vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(
        const dlib::matrix<dlib::rgb_pixel>& img) {
        // All this function does is make 100 copies of img, all slightly
        // jittered by being zoomed, rotated, and translated a little bit
        // differently. They are also randomly mirrored left to right.
        thread_local dlib::rand rnd;

        vector<dlib::matrix<dlib::rgb_pixel>> crops;
        for (int i = 0; i < 100; ++i)
            crops.push_back(dlib::jitter_image(img, rnd));

        return crops;
    }

    /** returns a map <person_id, score> for all the person whose face
     * descriptor(s) match (ie, distance < EUCLIDIAN_THRESHOLD) the provided
     * descriptor.
     *
     * The smaller the score, the better the match. As such, the best match can
     * be found with:
     *
     * ```cpp
     * auto scores = find_matches(descriptor);
     * auto id = min_element(scores.begin(), scores.end(),
     *                      [](decltype(scores)::value_type& l,
     *                         decltype(scores)::value_type& r)
     *                         -> bool { return l.second < r.second;}
     *                      )->first;
     * ```
     */
    map<Id, float> find_matches(Features descriptor) {
        map<Id, float> scores;

        for (const auto& kv : person_descriptors) {
            auto person_id = kv.first;
            for (const auto& known_descriptor : kv.second) {
                auto score = length(descriptor - known_descriptor);

                if (score < EUCLIDIAN_THRESHOLD) {
                    if (scores.count(person_id) and scores[person_id] > score) {
                        // new best match
                        scores[person_id] = score;
                    }

                    // we've got a match!
                    ROS_INFO_STREAM("Found a match with person " << person_id);
                    scores[person_id] = score;
                }
            }
        }

        return scores;
    }

   private:
    anet_type net;
    map<Id, vector<Features>> person_descriptors;
    map<Id, Id> face_person_map;
};
// ----------------------------------------------------------------------------------------

void onFace(hri::FaceWeakConstPtr face) {
    cout << "Received " << face.lock()->id() << endl;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "hri_face_identification");

    ros::NodeHandle nh;

    float match_threshold;
    ros::param::param<float>("/humans/face_identification/match_threshold",
                             match_threshold, 0.6);

    FaceRecogntion fr;

    ros::Rate loop_rate(10);

    HRIListener hri_listener;

    // hri_listener.onFace(&onFace);

    auto candidate_matches_pub =
        nh.advertise<hri_msgs::IdsMatch>("/humans/candidate_matches", 1, true);

    while (ros::ok()) {
        auto faces = hri_listener.getFaces();
        for (auto& f : faces) {
            auto face_id = f.first;
            auto face = f.second.lock();
            if (face) {
                if (face->aligned().empty()) continue;

                // wraps OpenCV image into a dlib image (no data copy)
                // ATTENTION: face->aligned() should not be modified while
                // wrapped: we first clone the image to ensure this does not
                // happen.
                // cv::Mat aligned_face(face->aligned().clone());
                cv::Mat aligned_face;
                cv::resize(face->aligned(), aligned_face, {IMG_SIZE, IMG_SIZE},
                           0, 0, cv::INTER_LINEAR);

                cout << "Got face " << face_id << endl;

                //                IplImage aligned_face_ipl = cvIplImage(
                //                    aligned_face);  // workaround for
                //                    older dlib --
                //                                    //
                //                                    https://github.com/davisking/dlib/issues/1949#issuecomment-580126950
                //                cv_image<bgr_pixel>
                //                dlib_face(&aligned_face_ipl);

                dlib::matrix<dlib::rgb_pixel> rgb_dlib_face;
                dlib::assign_image(
                    rgb_dlib_face,
                    dlib::cv_image<dlib::bgr_pixel>(cvIplImage(aligned_face)));
                // assign_image(rgb_dlib_face, dlib_face);

                //                vector<dlib::matrix<dlib::rgb_pixel>>
                //                dlib_faces;
                //               dlib_faces.push_back(rgb_dlib_face);

                //                vector<dlib::matrix<float, 0, 1>>
                //                face_descriptors =
                //                    net(dlib_faces);
            }
        }

        loop_rate.sleep();
        ros::spinOnce();
    }

    return 0;

    //    cout << "calculate face descriptors..." << endl;
    //    // This call asks the DNN to convert each face image in faces into
    //    a 128D
    //    // vector. In this 128D vector space, images from the same person
    //    will be
    //    // close to each other but vectors from different people will be
    //    far apart.
    //    // So we can use these vectors to identify if a pair of images are
    //    from the
    //    // same person or from different people.
    //    vector<dlib::matrix<float, 0, 1>> face_descriptors = net(faces);

    //    // It should also be noted that face recognition accuracy can be
    //    improved if
    //    // jittering is used when creating face descriptors.  In
    //    particular, to
    //    // get 99.38% on the LFW benchmark you need to use the
    //    jitter_image()
    //    // routine to compute the descriptors, like so:
    //    dlib::matrix<float, 0, 1> face_descriptor =
    //        mean(mat(net(jitter_image(faces[0]))));
    //    cout << "jittered face descriptor for one face: " <<
    //    trans(face_descriptor)
    //         << endl;
    //    // If you use the model without jittering, as we did when
    //    clustering the
    //    // bald guys, it gets an accuracy of 99.13% on the LFW benchmark.
    //    So
    //    // jittering makes the whole procedure a little more accurate but
    //    makes face
    //    // descriptor calculation slower.
}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

