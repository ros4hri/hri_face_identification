from functools import partial
import deepface

import cv2
import rospy
from hri_msgs.msg import IdsList
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

import pickle
import pathlib

from deepface import DeepFace
from deepface.commons.functions import find_input_shape
from tensorflow.keras.preprocessing import image

FACE_DB_DIRECTORY = pathlib.Path.home() / ".config/hri_face_identification"

# create the FACE_DB_DIRECTORY if it does not exist yet
FACE_DB_DIRECTORY.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.40  # for cosine distance


def cosine_distance(a, b):
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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
    img_pixels /= 255  # normalize input in [0, 1]

    return img_pixels


class FaceIdentification:
    def __init__(self):

        self.faces = {}

        self.identified_faces = {}
        self.load_face_database(FACE_DB_DIRECTORY)

        rospy.loginfo("Building Facenet model...")
        self.model = DeepFace.build_model("Facenet")
        print("Model input size: %s" % str(find_input_shape(self.model)))

        rospy.Subscriber("/humans/faces/tracked", IdsList, self.on_faces)

    def on_faces(self, data):

        known_faces = list(self.faces.keys())

        for face_id in data.ids:
            if face_id not in self.faces:
                rospy.loginfo("New face detected: %s", face_id)
                self.faces[face_id] = rospy.Subscriber(
                    "/humans/faces/%s/aligned" % face_id,
                    Image,
                    partial(FaceIdentification.on_face, self, face_id),
                )

        for id in known_faces:
            if id not in data.ids:
                del self.faces[id]

        # rospy.loginfo("Currently tracking %s faces" % len(self.faces))

    def on_face(self, face_id, img):
        # rospy.loginfo("Got a face image for face %s!" % face_id)

        img = CvBridge().imgmsg_to_cv2(img, desired_encoding="bgr8")

        # img = cv2.resize(find_input_shape(self.model))
        img = preprocess_face(img, find_input_shape(self.model))

        # image normalization
        mean, std = img.mean(), img.std()
        img = (img - mean) / std

        # project face onto pre-trained embedding
        embedding = self.model.predict(img)[0].tolist()

        candidates = self.find_candidates(embedding)

        if not candidates:
            rospy.loginfo("Can not recognise this face. Storing it as a new person.")
            self.store_face(face_id, embedding)
        else:
            if len(candidates) == 1:
                face_id, distance = candidates[0]
                rospy.loginfo(
                    "Found a match with ID %s (distance: %.2f)!" % (face_id, distance)
                )
            else:
                rospy.loginfo("Found more than one possible match:")
                for face_id, distance in candidates:
                    rospy.loginfo("  - %s (d=%.2f)" % (face_id, distance))

    def find_candidates(self, embedding):

        distances = {}

        for face_id, target_embedding in self.identified_faces.items():
            dst = cosine_distance(embedding, target_embedding)
            if dst < THRESHOLD:
                # keep the smallest distance for that face id
                if face_id not in distances or dst < distances[face_id]:
                    distances[face_id] = dst

        # returns candidates, sorted by increasing cosine distance
        return sorted(distances.items(), key=lambda x: x[1])

    def store_face(self, face_id, embedding):
        self.identified_faces[face_id] = embedding

    def load_face_database(self, path: pathlib.Path):

        db_path = path / "faces.db"

        if not db_path.exists():
            return

        with open(db_path, "rb") as f:
            self.identified_faces = pickle.load(f)

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
            % (len(self.identified_faces, db_path))
        )


if __name__ == "__main__":
    rospy.init_node("hri_face_identification")

    face_id = FaceIdentification()

    rospy.loginfo(
        "hri_face_identification ready. Waiting for face detections on /humans/faces/tracked"
    )
    rospy.spin()

    face_id.save_face_database(FACE_DB_DIRECTORY)
