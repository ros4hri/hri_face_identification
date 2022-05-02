#include <hri/face.h>
#include <hri/hri.h>
#include <hri_msgs/IdsMatch.h>
#include <ros/ros.h>

#include <opencv2/highgui.hpp>

#include "face_recognition.hpp"

using namespace ros;
using namespace hri;
using namespace std;

int main(int argc, char** argv) {
    ros::init(argc, argv, "hri_face_identification");

    ros::NodeHandle nh;

    float match_threshold;
    ros::param::param<float>("/humans/face_identification/match_threshold",
                             match_threshold, 0.6);

    FaceRecognition fr(match_threshold);

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

                cout << "Got face " << face_id << endl;
                fr.processFace(face->aligned());
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

