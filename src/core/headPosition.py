import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawind = mp.solutions.drawing_utils

drawing_spec = mp_drawind.DrawingSpec(thickness=1, circle_radius=1)


def detectFace(image: np.ndarray, draw=False):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False

    results = face_mesh.process(image)

    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape

    face_3d = []
    face_2d = []
    angle = 0
    face_pos_x_min = 999999
    face_pos_x_max = 0
    face_pos_y_min = 999999
    face_pos_y_max = 0
    detected = False

    if results.multi_face_landmarks:
        detected = True
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                # Finding position of face
                x, y = int(lm.x*img_w), int(lm.y * img_h)
                if x < face_pos_x_min:
                    face_pos_x_min = x
                if y < face_pos_y_min:
                    face_pos_y_min = y
                if x > face_pos_x_max:
                    face_pos_x_max = x
                if y > face_pos_y_max:
                    face_pos_y_max = y

                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (x, y)
                        nose_3d = (x, y, lm.z * 3000)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)

            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h/2], [0, focal_length, img_w/2], [0, 0, 1] ])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] - x*8))

            t1 = p2[1] - p1[1]
            t2 = p2[0] - p1[0]

            if t2 != 0 and (y >= 4  or y <= -4 or x >= 4 or x <= -4):
                if y > 0:
                    if x > 0:
                        angle = -1 * (math.atan(t1 / t2) * 360 / (2 * math.pi))
                    else:
                        angle = 360 - (math.atan(t1 / t2) * 360 / (2 * math.pi))
                else:
                        angle = 180 - (math.atan(t1 / t2) * 360 / (2 * math.pi))


            cv2.line(image, p1, p2, (255, 0, 0), 3)

        if draw:
            mp_drawind.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)

    return image, angle, detected, (face_pos_x_min, face_pos_x_max), (face_pos_y_min, face_pos_y_max)



from typing import Tuple, Union
import math

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image


# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

def detectFace2(image):
# STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

# STEP 4: Detect faces in the input image.
    detection_result = detector.detect(image)
    return detection_result



