^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package hri_face_identification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.1.1 (2022-05-02)
------------------

Initial release in C++, based on dlib face recognition pipeline.
http://dlib.net/dnn_face_recognition_ex.cpp.html

Main features:

* publish candidate match under /humans/candidate_matches
* add additional face descriptors to people whose recognition's score is average
* store/load the face database as a json file using nlohmann's JSON C++ library
* configurable matching threshold
* Full ROS packaging
* Contributors: Séverin Lemaignan
