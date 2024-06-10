import numpy as np
import cv2
import mediapipe as mp
import time
from pydub import AudioSegment
from pydub.playback import play
import logging
import os
import contextlib
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'

def suppress_stderr():
    return contextlib.redirect_stderr(open(os.devnull, 'w'))

# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Thresholds for head pose detection
THRESHOLD_LEFT_RIGHT = 15  # Degrees for detecting left or right
THRESHOLD_DOWN = 10  # Degrees for detecting down
MAX_WARNINGS = 10  # Maximum number of warnings before exit

# Load warning sound
# WARNING_SOUND = AudioSegment.from_wav('warning.wav')

# Warning types
NO_FACE = 1
MULTIPLE_FACES = 2
LOOKING_AWAY = 3

def is_cheating(image):
    with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
        
        if num_faces > 1:
            print("2 faces detected...")
            return MULTIPLE_FACES
        elif num_faces == 0:
            print("Please show your face...")
            return NO_FACE
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_2d = []
                face_3d = []
                img_h, img_w, img_c = image.shape

                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                       [0, focal_length, img_h / 2],
                                       [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

                rmat, jac = cv2.Rodrigues(rotation_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if y < -THRESHOLD_LEFT_RIGHT or y > THRESHOLD_LEFT_RIGHT or x < -THRESHOLD_DOWN:
                    print("Please look at the screen...")
                    return LOOKING_AWAY
        
    return None

def capture_and_check(max_warnings=MAX_WARNINGS):
    cap = cv2.VideoCapture(0)
    warning_count = 0
    consecutive_look_away_count = 0

    # Initialize timer
    last_capture_time = time.time()

    while cap.isOpened():
        current_time = time.time()
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Capture image every 5 seconds
        if current_time - last_capture_time >= 5:
            last_capture_time = current_time
            print("Sending image to check...........")
            
            # Save the captured image as JPEG
            timestamp = int(time.time())
            filename = f"captured_image_{timestamp}.jpg"
            cv2.imwrite(filename, image)
            
            warning_type = is_cheating(image)
            if warning_type == LOOKING_AWAY:
                consecutive_look_away_count += 1
                if consecutive_look_away_count >= 2:
                    print("Warning issued for looking away...")
                    warning_count += 1
                    consecutive_look_away_count = 0  # Reset the consecutive look away count after issuing a warning
            else:
                consecutive_look_away_count = 0  # Reset if the user looks back at the screen

            if warning_type == NO_FACE or warning_type == MULTIPLE_FACES:
                print(f"Warning for {warning_type}")
                warning_count += 1

            if warning_count >= max_warnings:
                print("Too many warnings. Exiting...")
                break

        resized_frame = cv2.resize(image, (800, 600))
        cv2.imshow("Face Mesh", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    with suppress_stderr():
        capture_and_check()