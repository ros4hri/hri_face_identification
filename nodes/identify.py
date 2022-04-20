from functools import partial
import deepface

import cv2
import rospy
from hri_msgs.msg import IdsList
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

from deepface import DeepFace
from deepface.commons.functions import find_input_shape
from tensorflow.keras.preprocessing import image


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

        rospy.loginfo("Currently tracking %s faces" % len(self.faces))

    def on_face(self, face_id, img):
        rospy.loginfo("Got a face image for face %s!" % face_id)

        img = CvBridge().imgmsg_to_cv2(img, desired_encoding="bgr8")

        # img = cv2.resize(find_input_shape(self.model))
        img = preprocess_face(img, find_input_shape(self.model))

        # image normalization
        mean, std = img.mean(), img.std()
        img = (img - mean) / std

        # project face onto pre-trained embedding
        embedding = self.model.predict(img)[0].tolist()

        print(embedding)


if __name__ == "__main__":
    rospy.init_node("face_identification_deepface")

    face_id = FaceIdentification()

    rospy.loginfo(
        "hri_face_identification_deepface ready. Waiting for faces to be detected"
    )
    rospy.spin()
