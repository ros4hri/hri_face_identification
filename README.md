hri_face_identification
=======================

A [ROS4HRI](https://wiki.ros.org/hri)-compatible face identification package.
It is built on top of
[dlib's face recognition pipeline](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html).
It performs face recognition at > 100fps on a GTX10xx generation mobile GPU.

Because this node is meant to be used in a broader face processing pipeline,
it does not perform face detection itself.
It expects instead faces to be detected and published under `/humans/faces/...`
(see below the list of subscribed topics).

## ROS API

### Parameters

All parameters are loaded in the lifecycle `configuration` transition.

- `model_path` (default: `<pkg share>/dlib_face_recognition_resnet_model_v1.dat`):
  Absolute path to the trained dlib resnet face identification model to be loaded.
- `match_threshold` (default: 0.5):
  Distance threshold (in the face embedding space) to consider two faces to belong to the same person.
- `face_database_paths` (default: no database):
  A list of full paths to the face databases where known faces will be loaded from.
  The first one will also be used to permanently store newly detected faces.
  Delete these files to 'start from scratch', with no known faces.
- `can_learn_new_faces` (default: `true`):
  Whether or not unknown faces will be added to the database.
  If set to false, only *previously identified* faces are recognised (and the face database will not be modified).
- `identify_all_faces` (default: `false`):
  Whether or not already-tracked faces are re-identified.
  If false, face identification is only performed when face id changes,
  i.e. a currently-tracked face will not be re-identified every frame.
  If true, all received faces (on `/humans/faces/<id>/aligned`) will run through face identification at every frame.
- `processing_period` (int, default: 100):
  Best effort period for processing input images in milliseconds.

### Topics

This package follows the ROS4HRI conventions ([REP-155](https://www.ros.org/reps/rep-0155.html)).
If the topic message type is not indicated, the ROS4HRI convention is implied.

#### Subscribed

- `/humans/faces/tracked`
- `/humans/faces/<face_id>/aligned`

#### Published

- `/humans/candidate_matches`:
  Correspondances between face IDs and (recognised) person IDs (alongside with a confidence level). 
  The `person_id` IDs are randomly generated when a new, unknown, face is detected.

## Examples

### Identify faces

For an example of usage, execute in different terminals:
- USB camera:
  1. `apt install ros-humble-usb-cam`
  2. `ros2 run usb_cam usb_cam_node_exe`
- HRI face detect:
  1. Either
    - if you are on a PAL robot `apt install ros-humble-hri-face-detect`
    - otherwise build and install from [source](https://github.com/ros4hri/hri_face_detect).
  2. `ros2 launch hri_face_detect face_detect.launch.py`
- HRI face identification:
  1. `apt install ros-humble-hri-face-identification`
  2. `ros2 launch hri_face_identification face_identification.launch.py`
- RViz with HRI plugin:
  1. `apt install ros-humble-rviz2`
  1. `apt install ros-humble-hri-rviz`
  2. `rviz2`

In RViz, add the 'Humans' plugin to see the detected faces.
The face IDs should be permanently assigned to the same people.

### Offline face database creation

Execute in different terminals:
- USB camera:
  1. `apt install ros-humble-usb-cam`
  2. `ros2 run usb_cam usb_cam_node_exe`
- HRI face detect:
  1. Either:
    - if you are on a PAL robot `apt install ros-humble-hri-face-detect`.
    - otherwise build and install from [source](https://github.com/ros4hri/hri_face_detect).
  2. `ros2 launch hri_face_detect face_detect.launch.py`
- HRI face identification recorder:
  1. `apt install ros-humble-hri-face-identification`
  2. For each person:
    - `ros2 run hri_face_identification db_record -n <person_name>` (use `--help` for additional arguments).
    - A camera image window should appear; selecting it use SPACE to start/stop recording a scene, ESC to terminate.
    - Record multiple scenes, each with at least 30 images and in a different lightning condition.
  3. The images are saved by default in `/tmp/face_dataset`

After the recordings are completed, the processed faces database is obtained executing
`ros2 run hri_face_identification db_process` (use `--help` for additional arguments).
By default it will read the persons facs images database from `/tmp/face_dataset` and
output a `/tmp/face_dataset/face_db.json` containing the anonymized face embeddings.
This file can be passed to the `hri_face_identification` node, along with others, through the `face_database_paths` parameter.
Check the terminal output to see the anonymous id mappings.
