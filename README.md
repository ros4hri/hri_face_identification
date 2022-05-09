hri_face_identification
=======================

Overview
--------

`hri_face_identification` is a [ROS4HRI](https://wiki.ros.org/hri)-compatible
face identification node.

It is built on top of [dlib's face recognition
pipeline](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html).

It performs face recognition at > 100fps on a GTX10xx generation mobile GPU.

Because this node is meant to be used in a broader face processing pipeline, it
does not perform face detection itself. It expects instead faces to be detected
and published under `/humans/faces/...` (see below the list of subscribed topics).

ROS API
-------

### Parameters

- `/humans/face_identification/match_threshold` (default: 0.6): distance
  threshold (in the face embedding space) to consider two faces to belong to the
  same person.
- `/humans/face_identification/face_database_path` (default: `face_db.json`):
  full path to the face database where known faces will be permanently stored.
  Delete this file to 'start from scratch', with no known faces.
- `~can_learn_new_faces` (default: `true`):
  whether or not unknown faces will be added to the database. If set to false,
  only *previously identified* faces are recognised (and the face database will
  not be modified).
- `~identify_all_faces` (default: `false`):
  whether or not already-tracked faces are re-identified. If false, face
  identification is only performed when face id changes, ie a currently-tracked
  face will not be re-identified every frame; if true, all received faces (on
  `/humans/faces/<id>/aligned`) will run through face identification at every
  frame.

### Topics

`hri_face_identification` follows the ROS4HRI conventions (REP-155).

#### Subscribed topics

- `/humans/faces/tracked`
  ([hri_msgs/IdsList](http://docs.ros.org/en/api/hri_msgs/html/msg/IdsList.html)):
  list of the faces currently detected.
- `/humans/faces/<face_id>/aligned`
  ([sensor_msgs/Image](http://docs.ros.org/en/api/sensor_msgs/html/msg/Image.html)):
  the aligned faces to recognise. Note that internally, the faces are always
  resized to 150x150px before running the recognition.

#### Published topics

- `/humans/candidate_matches`
  ([hri_msgs/IdsMatch](http://docs.ros.org/en/api/hri_msgs/html/msg/IdsMatch.html)):
  correspondances between face IDs and (recognised) person IDs (alongside with a
  confidence level). The `person_id` IDs are randomly generated when a new,
  unknown, face is detected.



