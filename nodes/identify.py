from functools import partial
import threading
from queue import Queue
import time
import uuid
import pickle
import pathlib
from threading import Thread
import numpy as np

import cv2

import rospy
from hri_msgs.msg import IdsList, IdsMatch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import deepface
from deepface import DeepFace
from deepface.commons.functions import find_input_shape
from tensorflow.keras.preprocessing import image

FACE_DB_DIRECTORY = pathlib.Path.home() / ".config/hri_face_identification"

# create the FACE_DB_DIRECTORY if it does not exist yet
FACE_DB_DIRECTORY.mkdir(parents=True, exist_ok=True)

MODEL = "Facenet"  # ~2.2GB of VRAM, ~50ms inference time
# MODEL = "DeepFace" # unconvincing results + slow
# MODEL = "ArcFace"
# MODEL = "DeepID"

# distance threshold below which 2 faces are considered as belonging to the
# same person
if MODEL == "Facenet":
    THRESHOLD = 0.40  # for cosine distance
elif MODEL == "ArcFace":
    THRESHOLD = 0.68  # for cosine distance
elif MODEL == "DeepFace":
    THRESHOLD = 0.23  # for cosine distance
elif MODEL == "DeepID":
    THRESHOLD = 0.015  # for cosine distance


# if the distance of a face to a known face is between
# THRESHOLD_ADDITIONAL_FACE and THRESHOLD, that face will be added to the
# database as a another exemplar for the same person
THRESHOLD_ADDITIONAL_FACE = THRESHOLD / 2  # for cosine distance


DEBUG = True


def cosine_distance(a, b):
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def confidence_from_distance(distance):

    # for now, a linear mapping to the distance. Might explore other function in the future
    return 1.0 - (distance / THRESHOLD)


def preprocess_face(img, target_size=(224, 224), grayscale=False):

    # post-processing
    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize image to expected shape

    # img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        if grayscale == False:
            # Put the base image in the middle of the padded image
            img = np.pad(
                img,
                (
                    (diff_0 // 2, diff_0 - diff_0 // 2),
                    (diff_1 // 2, diff_1 - diff_1 // 2),
                    (0, 0),
                ),
                "constant",
            )
        else:
            img = np.pad(
                img,
                (
                    (diff_0 // 2, diff_0 - diff_0 // 2),
                    (diff_1 // 2, diff_1 - diff_1 // 2),
                ),
                "constant",
            )

    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    # ---------------------------------------------------

    # normalizing the image pixels

    img_pixels = image.img_to_array(img)  # what this line doing? must?
    img_pixels = np.expand_dims(img_pixels, axis=0)
    # img_pixels /= 255  # normalize input in [0, 1]

    return img_pixels


class FaceIdentification:

    last_id = 1

    def __init__(self, deterministic_id=False):

        self.is_shutting_down = False
        self.deterministic_id = deterministic_id

        self.faces = {}

        self.incoming_face = Queue(maxsize=1)

        self.identified_faces = {}
        self.load_face_database(FACE_DB_DIRECTORY)

        rospy.loginfo("Building %s model..." % MODEL)
        self.model = DeepFace.build_model(MODEL)
        print("Model input size: %s" % str(find_input_shape(self.model)))

        rospy.Subscriber("/humans/faces/tracked", IdsList, self.on_faces)

        self.matches_pub = rospy.Publisher(
            "/humans/candidate_matches", IdsMatch, queue_size=1
        )

        self.processing_thread = Thread(target=self.process_face)
        self.processing_thread.start()

    def close(self):
        rospy.loginfo("Stopping face identification...")

        self.is_shutting_down = True

        for _, sub in self.faces.items():
            sub.unregister()

        self.processing_thread.join(0.5)

        rospy.loginfo("Stopped faces identification.")
        rospy.sleep(
            0.1
        )  # ensure the last messages published in this method (detector.close) are effectively sent.

    def on_faces(self, data):

        known_faces = list(self.faces.keys())

        for face_id in data.ids:
            if face_id not in self.faces:
                rospy.loginfo("New face detected: %s", face_id)
                self.faces[face_id] = rospy.Subscriber(
                    "/humans/faces/%s/aligned" % face_id,
                    Image,
                    partial(FaceIdentification.on_face, self, face_id),
                    queue_size=1,
                )

        for id in known_faces:
            if id not in data.ids:
                self.faces[id].unregister()
                del self.faces[id]

        # rospy.loginfo("Currently tracking %s faces" % len(self.faces))

    def on_face(self, face_id, img):

        # still processing the previous face? drop new one
        if self.incoming_face.full():
            return

        self.incoming_face.put((face_id, img))

    def process_face(self):

        while not self.is_shutting_down:

            face_id, ros_img = self.incoming_face.get()

            # rospy.loginfo("Got a face image for face %s!" % face_id)

            start = time.time()

            raw_img = CvBridge().imgmsg_to_cv2(ros_img, desired_encoding="bgr8")

            # img = cv2.resize(find_input_shape(self.model))
            img = preprocess_face(raw_img, find_input_shape(self.model))

            # image normalization; code from DeepFace's functions.py
            if MODEL == "Facenet":
                mean, std = img.mean(), img.std()
                img = (img - mean) / std

            elif MODEL == "VGGFace":
                # mean subtraction based on VGGFace1 training data
                img[..., 0] -= 93.5940
                img[..., 1] -= 104.7624
                img[..., 2] -= 129.1863

            elif MODEL == "VGGFace2":
                # mean subtraction based on VGGFace2 training data
                img[..., 0] -= 91.4953
                img[..., 1] -= 103.8827
                img[..., 2] -= 131.0912

            elif MODEL == "ArcFace":
                # Reference study: The faces are cropped and resized to 112×112,
                # and each pixel (ranged between [0, 255]) in RGB images is normalised
                # by subtracting 127.5 then divided by 128.
                img -= 127.5
                img /= 128

            elif MODEL == "DeepID":
                img /= 255

            # project face onto pre-trained embedding
            embedding = self.model.predict(img)[0].tolist()

            candidates = self.find_candidates(embedding)

            if not candidates:
                person_id = str(uuid.uuid4())[:5]  # for a 5 char long ID
                # generate unique ID
                if self.deterministic_id:
                    person_id = "person_%04d" % self.last_id
                    self.last_id = (self.last_id + 1) % 10000
                else:
                    person_id = str(uuid.uuid4())[:5]  # for a 5 char long ID

                rospy.loginfo(
                    "Can not recognise this face. Storing it as new person <%s>."
                    % person_id
                )

                self.store_face(person_id, embedding)
            else:
                if len(candidates) == 1:
                    person_id, distance = candidates[0]
                    rospy.loginfo(
                        "Found a match with ID %s (distance: %.2f)!"
                        % (person_id, distance)
                    )

                    ids_match = IdsMatch()
                    ids_match.face_id = face_id
                    ids_match.person_id = person_id
                    ids_match.confidence = confidence_from_distance(distance)

                    self.matches_pub.publish(ids_match)

                    if distance > THRESHOLD_ADDITIONAL_FACE:
                        self.store_face(person_id, embedding)
                else:
                    rospy.loginfo("Found more than one possible match:")
                    for person_id, distance in candidates:
                        rospy.loginfo("  - %s (d=%.2f)" % (person_id, distance))

                        ids_match = IdsMatch()
                        ids_match.face_id = face_id
                        ids_match.person_id = person_id
                        ids_match.confidence = confidence_from_distance(distance)

                        self.matches_pub.publish(ids_match)

            rospy.loginfo("Recognition took %.1fms" % ((time.time() - start) * 1000))

            if DEBUG:
                cv2.imshow("Face recognition with %s" % MODEL, raw_img)
                cv2.waitKey(1)

            # have we received a new face while processing the current one? if so, discard it,
            # as it is probably outdated by now. We will wait for the new one in the next loop
            if self.incoming_face.full():
                self.incoming_face.get()

    def find_candidates(self, embedding):

        distances = {}

        for person_id, target_embeddings in self.identified_faces.items():
            for target_embedding in target_embeddings:
                dst = cosine_distance(embedding, target_embedding)
                if dst < THRESHOLD:
                    # keep the smallest distance for that face id
                    if person_id not in distances or dst < distances[person_id]:
                        distances[person_id] = dst

        # returns candidates, sorted by increasing cosine distance
        return sorted(distances.items(), key=lambda x: x[1])

    def store_face(self, person_id, embedding):
        self.identified_faces.setdefault(person_id, []).append(embedding)

    def load_face_database(self, path: pathlib.Path):

        db_path = path / "faces.db"

        if not db_path.exists():
            return

        with open(db_path, "rb") as f:
            self.identified_faces = pickle.load(f)

        self.last_id = len(self.identified_faces) + 1

        rospy.loginfo(
            "Loaded %s face ids from stored in %s."
            % (len(self.identified_faces), db_path)
        )

    def save_face_database(self, path: pathlib.Path):

        db_path = path / "faces.db"

        with open(db_path, "wb") as f:
            pickle.dump(self.identified_faces, f)

        rospy.loginfo(
            "%s face ids stored in %s. Delete this file to erase all known identities."
            % (len(self.identified_faces), db_path)
        )


if __name__ == "__main__":
    rospy.init_node("hri_face_identification")

    face_id = FaceIdentification(deterministic_id=True)

    rospy.loginfo(
        "hri_face_identification ready. Waiting for face detections on /humans/faces/tracked"
    )

    rospy.on_shutdown(face_id.close)
    rospy.spin()

    face_id.save_face_database(FACE_DB_DIRECTORY)
