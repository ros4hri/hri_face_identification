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

#include <functional>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <diagnostic_msgs/DiagnosticStatus.h>
#include <diagnostic_updater/diagnostic_updater.h>
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

    bool create_person_if_needed;
    ros::param::param<bool>("~can_learn_new_faces", create_person_if_needed,
                            true);

    if (create_person_if_needed) {
        ROS_INFO(
            "_can_learn_new_faces:=true (default): I will learn new faces, and "
            "add them to the face database");
    } else {
        ROS_WARN(
            "_can_learn_new_faces:=false: I will NOT learn new faces; only "
            "people already in the database will be recognised");
    }

    // if false, face identification is only performed when face id changes, ie
    // a currently tracked face will not be re-identified every frame; if true,
    // all received faces (on /humans/faces/<id>/aligned) will run through face
    // identification.
    bool reidentify_all_faces;
    ros::param::param<bool>("~identify_all_faces", reidentify_all_faces, false);

    vector<string> face_db_paths;
    ros::param::param<vector<string>>(
        "/humans/face_identification/face_database_paths", face_db_paths,
        {"face_db.json"});

    FaceRecognition fr(match_threshold);

    for (const auto& db : face_db_paths) {
        fr.loadFaceDB(db);
    }

    ros::Rate loop_rate(10);

    HRIListener hri_listener;

    auto candidate_matches_pub = nh.advertise<hri_msgs::IdsMatch>(
        "/humans/candidate_matches", 10, false);

    hri_listener.onFace(&onFace);

    // add diagnostics
    diagnostic_updater::Updater diag_updater{nh, ros::NodeHandle("~"),
                                             " Social perception: Face analysis"}; // adding initial space in 'node_name' string since diagnostic_updater removes the first char
    diagnostic_updater::CompositeDiagnosticTask diag_composite_task{"Identification"};
    diag_updater.setHardwareID("none");
    diag_updater.add(diag_composite_task);

    diagnostic_updater::FunctionDiagnosticTask diag_base_task("base_task", 
        [&face_db_paths](diagnostic_updater::DiagnosticStatusWrapper& status){
            status.summary(diagnostic_msgs::DiagnosticStatus::OK, "OK");
            status.add("Package name", "hri_face_identification");
            std::ostringstream face_db_paths_string{};
            const char* const delim = ", ";
            std::copy(face_db_paths.begin(), face_db_paths.end(), 
                      std::ostream_iterator<std::string>(face_db_paths_string, delim));
            status.add("Face database paths", face_db_paths_string.str());
            status.add("Currently detected faces", tracked_faces.size());
        });
    diag_composite_task.addTask(&diag_base_task);

    diagnostic_updater::FunctionDiagnosticTask diag_face_recognition_task("face_recognition_task",
        std::bind(&FaceRecognition::doDiagnostics, &fr, std::placeholders::_1));   
    diag_composite_task.addTask(&diag_face_recognition_task);

    // ready to go!
    semaphore_pub.publish(std_msgs::Empty());

    // mapping between a face_id and all the possible recognised person_id
    // (with their confidence level) for that face.
    map<Id, map<Id, float>> face_persons_map;

    while (ros::ok()) {
        vector<Id> faces_to_remove;

        for (const auto& kv : tracked_faces) {
            auto face_id = kv.first;
            auto face = kv.second.lock();
            if (face) {
                if (face->aligned().empty()) continue;

                ROS_DEBUG_STREAM("Got face " << face_id);

                map<Id, float> results;

                if (!reidentify_all_faces &&
                    face_persons_map.count(face_id) != 0) {
                    results = face_persons_map[face_id];
                } else {
                    ROS_INFO("Trying to identify the face...");
                    // note that this might return more than one match! each
                    // match has an associated confidence level
                    results = fr.getAllMatches(face->aligned(),
                                               create_person_if_needed);
                }

                for (const auto& res : results) {
                    hri_msgs::IdsMatch match;
                    match.id1 = res.first;
                    match.id1_type = hri_msgs::IdsMatch::PERSON;
                    match.confidence = res.second;
                    match.id2 = face_id;
                    match.id2_type = hri_msgs::IdsMatch::FACE;

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
                    match.id1 = person.first;
                    match.id1_type = hri_msgs::IdsMatch::PERSON;
                    match.confidence = 0.0;
                    match.id2 = face_id;
                    match.id2_type = hri_msgs::IdsMatch::FACE;

                    candidate_matches_pub.publish(match);
                }
            }
        }

        for (const auto& id : faces_to_remove) {
            tracked_faces.erase(id);
        }

        diag_updater.update();
        loop_rate.sleep();
        ros::spinOnce();
    }

    fr.storeFaceDB(face_db_paths[0]);
    cout << "Bye bye!" << endl;

    return 0;
}

