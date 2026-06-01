import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import math

# joint pairs , angles
JOINT_CONSTANTS = {
        "1": [("12-14", "14-16"), (-90)]
        }

ANGLE_THRESHOLD = 15 # degrees

def make_decision(calc_angles: list):
    for sign in JOINT_CONSTANTS.keys():
        if (check_requirements(JOINT_CONSTANTS[sign], calc_angles)):
            print("Sign predicted!")
            return sign
    return "Waiting"

def check_requirements(req: list, calc_angles: list) -> bool:
    if isinstance(req[1], int):
        angle = req[1]

    conds : bool = []
    for pair in req[0]:
        try:
            conds.append(calc_angles[pair] < (angle + ANGLE_THRESHOLD) and calc_angles[pair] > (angle - ANGLE_THRESHOLD))
            print(f"ANGLES {pair}: {calc_angles[pair]}\n")
        except:
            print("Angles not found")
            return False

    if all(conds):
        return True
    else:
        return False



main_joint_idxs = [12, 14, 16, 18, 20, 22, 11, 13, 15, 17, 19, 21]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

predicted_gesture = "Waiting"

print("Starting live prediction loop...")
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
        
    # Process with MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    current_angles = {}
    if results.pose_landmarks:
        landmark_list = results.pose_landmarks.landmark
        for idx, landmark in enumerate(landmark_list):
            # current_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            if idx in main_joint_idxs:
                if idx == 22 or idx == 21: continue

                next_landmark = landmark_list[idx+2]
                diff_y = next_landmark.y - landmark.y
                diff_x = next_landmark.x - landmark.x
                angle = math.degrees(math.atan(diff_y/diff_x))
                current_angles[f"{idx}-{idx+2}"] = angle

                # hand joints
                if idx == 16 or 15:

                    next_landmark = landmark_list[idx+4]
                    diff_y = next_landmark.y - landmark.y
                    diff_x = next_landmark.x - landmark.x
                    angle = math.degrees(math.atan(diff_y/diff_x))
                    current_angles[f"{idx}-{idx+4}"] = angle

    else:
        print("Person is lost")
        # current_landmarks = [0.0] * (33 * 4) # Fallback if person is lost

    predicted_gesture = make_decision(current_angles)
    # simple logic (1 sign)
    try:
        # print(f"ANGLES 12-14: {current_angles['12-14']}\n")
        print(f"ANGLES 14-16: {current_angles['14-16']}\n")
    except:
        print("Bibki")
     

        
    cv2.putText(frame, predicted_gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('Pi Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
