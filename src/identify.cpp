#include <hri/face.h>
#include <hri/hri.h>
#include <hri_msgs/IdsMatch.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>

#include <opencv2/highgui.hpp>

#include "face_recognition.hpp"

using namespace ros;
using namespace hri;
using namespace std;

int main(int argc, char** argv) {
    ros::init(argc, argv, "hri_face_identification");

    ros::NodeHandle nh;
    auto semaphore_pub = nh.advertise<std_msgs::Empty>(
        "/hri_face_identification/ready", 1, true);

    float match_threshold;
    ros::param::param<float>("/humans/face_identification/match_threshold",
                             match_threshold, 0.6);

    string face_db_path;
    ros::param::param<string>("/humans/face_identification/face_database_path",
                              face_db_path, "face_db.json");

    FaceRecognition fr(match_threshold);

    fr.loadFaceDB(face_db_path);

    ros::Rate loop_rate(10);

    HRIListener hri_listener;

    // hri_listener.onFace(&onFace);

    auto candidate_matches_pub =
        nh.advertise<hri_msgs::IdsMatch>("/humans/candidate_matches", 1, false);

    // ready to go!
    semaphore_pub.publish(std_msgs::Empty());

    while (ros::ok()) {
        auto faces = hri_listener.getFaces();
        for (auto& f : faces) {
            auto face_id = f.first;
            auto face = f.second.lock();
            if (face) {
                if (face->aligned().empty()) continue;

                ROS_INFO_STREAM("Got face " << face_id);
                auto results = fr.processFace(face->aligned());

                for (const auto& res : results) {
                    hri_msgs::IdsMatch match;
                    match.person_id = res.first;
                    match.confidence = res.second;
                    match.face_id = face_id;

                    candidate_matches_pub.publish(match);
                }
            }
        }

        loop_rate.sleep();
        ros::spinOnce();
    }

    fr.storeFaceDB(face_db_path);
    cout << "Bye bye!" << endl;

    return 0;
}

