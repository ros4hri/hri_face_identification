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

map<Id, FaceWeakConstPtr> tracked_faces;

void onFace(FaceWeakConstPtr face) {
    auto face_ptr = face.lock();
    if (face_ptr) {
        tracked_faces[face_ptr->id()] = face;
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "hri_face_identification");

    ros::NodeHandle nh;
    auto semaphore_pub = nh.advertise<std_msgs::Empty>(
        "/hri_face_identification/ready", 1, true);

    float match_threshold;
    ros::param::param<float>("/humans/face_identification/match_threshold",
                             match_threshold, DEFAULT_MATCH_THRESHOLD);

    string face_db_path;
    ros::param::param<string>("/humans/face_identification/face_database_path",
                              face_db_path, "face_db.json");

    FaceRecognition fr(match_threshold);

    fr.loadFaceDB(face_db_path);

    ros::Rate loop_rate(10);

    HRIListener hri_listener;

    auto candidate_matches_pub =
        nh.advertise<hri_msgs::IdsMatch>("/humans/candidate_matches", 1, false);

    hri_listener.onFace(&onFace);

    // ready to go!
    semaphore_pub.publish(std_msgs::Empty());

    map<Id, map<Id, float> > face_persons_map;

    while (ros::ok()) {
        vector<Id> faces_to_remove;

        for (const auto& kv : tracked_faces) {
            auto face_id = kv.first;
            auto face = kv.second.lock();
            if (face) {
                if (face->aligned().empty()) continue;

                ROS_INFO_STREAM("Got face " << face_id);
                auto results = fr.processFace(face->aligned());

                for (const auto& res : results) {
                    hri_msgs::IdsMatch match;
                    match.person_id = res.first;
                    match.confidence = res.second;
                    match.face_id = face_id;

                    face_persons_map[face_id][res.first] = res.second;

                    candidate_matches_pub.publish(match);
                }
            }
            // face.lock() returns an empty pointer? the face does not exist
            // anymore!
            else {
                faces_to_remove.push_back(face_id);

                // for all the person id previously associated to this face,
                // publish a 'match' with confidence = 0 to dis-associate them.
                for (const auto& person : face_persons_map[face_id]) {
                    ROS_INFO_STREAM(
                        "Face " << face_id
                                << " not tracked. Dis-associating from person "
                                << person.first);

                    hri_msgs::IdsMatch match;
                    match.person_id = person.first;
                    match.confidence = 0.0;
                    match.face_id = face_id;

                    candidate_matches_pub.publish(match);
                }
            }
        }

        for (const auto& id : faces_to_remove) {
            tracked_faces.erase(id);
        }

        loop_rate.sleep();
        ros::spinOnce();
    }

    fr.storeFaceDB(face_db_path);
    cout << "Bye bye!" << endl;

    return 0;
}

