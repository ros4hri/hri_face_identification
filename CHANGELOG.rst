^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package hri_face_identification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.2.0 (2024-07-29)
------------------
* Paps7 conformant
* Contributors: lorenzoferrini

2.1.0 (2024-04-25)
------------------
* add pal module
* update privacy msg class name
* faces databases from parameters are appended to the resource ones
* send data management notice messages
* rework face databases api
* fix shutdown execution on SIGINT
* load default face database from the resource index
* port the database creation helper executables
* remove unknown face from db
* Contributors: Luka Juricic

2.0.0 (2024-02-07)
------------------
* port tests to humble
* extract test images from rosbags
* port node to humble; remove batch_process.cpp
* change license to Apache 2.0
* change folder structure
* use the correct param name for match threshold
* Contributors: Luka Juricic, Séverin Lemaignan

0.3.6 (2023-05-12)
------------------
* improve diagnostic message
* - address review issues
  - split diagnostics into the relevant modules
* add basic diagnostics
* Contributors: Luka Juricic

0.3.5 (2023-01-12)
------------------
* add a launch file pre-configure to always load the test face database
* Contributors: Séverin Lemaignan

0.3.4 (2023-01-11)
------------------
* add a pre-trained face database, with 2 faces ('John' and 'Jane')
* make it possible to load several face databases
  Required to provide a 'static' test database, as well as a customer-specific database
* add utility to batch-process faces and export features to CSV
* Contributors: Séverin Lemaignan

0.3.2 (2022-06-01)
------------------
* update to new IdsMatch.msg
* increase candidate_matches queue size
  This helps hri_person_manager not to miss important messages like 'X disappeared'
* Contributors: Séverin Lemaignan

0.3.1 (2022-05-10)
------------------
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
