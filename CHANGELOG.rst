^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package hri_face_identification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* [api] polish API
  renamed 'processFaces' into 'getAllMatches' and 'bestMatch' into 'getBestMatch'
* Contributors: Séverin Lemaignan

0.3.0 (2022-05-09)
------------------
* add option to only identify faces when their (tracking) id changes
  This is the new default, significantly reducing the CPU/GPU needs for the node.
* always generate different random IDs, even if less than a sec since last generation
  (previous seed was in sec, henceforth generating same IDs within the same second)
* add method to empty to list of known faces (dropFaceDB)
* when a face disappear, publish a 'match' of confidence 0 to disassociate the face/person
* significantly expand the unit-tests, with a much broader set of faces
* only create new faces in db if requested (flag create_id_if_needed). By default, add new faces to the face database
* add bestMatch to directly return best candidate person id
* generate person_id with only letters to avoid purely numerical id (that would be invalid ROS id)
* Contributors: Séverin Lemaignan

0.2.0 (2022-05-02)
------------------
* add LICENSE (BSD) 
* be explicit about BLAS/LAPACK, due an issue/bug in the way old versions of dlib includes it
* add initial unit test
* install and properly access the dlib RESnet pre-trained weights
* publish an empty msg on /hri_face_identification when ready to proceed
* [doc] add README with API documentation
* Contributors: Séverin Lemaignan

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
