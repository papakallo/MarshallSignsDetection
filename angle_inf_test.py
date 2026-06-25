import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import math
from gpiozero import LED

LEDS = {
    "green": LED(14),
    "red": LED(21),
    "blue": LED(18),
    "yellow": LED(23),
    "red2": LED(20)
}

# joint pairs , angles
JOINT_CONSTANTS = {
        "Wingwalker/Guide": [("12-14", "14-16", "11-13", "13-15"), (-90, -90, 90, 90), ["yellow", "red"]],
        #"2": [("12-14", "14-16", "16-22"), (180, 180, -160), ["red"]], # 2 and 3 are very tricky, thumb should be at 90 but reasons (still very buggy)
        #"3": [("12-14", "14-16", "16-22"), (180, 180, 160), ["blue"]],
        "TurnLeft": [("12-14", "14-16", "11-13", "13-15"), (180, -90, 0, 0), ["blue", "yellow"]],
        "TurnRight": [("12-14", "14-16", "11-13", "13-15"), (180, 180, 0, -90), ["green", "yellow"]],
        #"Fire": [("12-14", "14-16"), (-120, -115), ["blue", "yellow"]], # values picked blindly (good) fire
        "Stop": [("12-14", "14-16", "11-13", "13-15"), (-90, -45, -90, -135), ["red", "red2"]],
        "IdentifyGate": [("12-14", "14-16", "11-13", "13-15"), (-90, -90, -90, -90), ["red", "blue"]], 
        "ProceedToNextSignalManLeft": [("14-16", "11-13", "13-15"), (0, 0, 0), ["blue", "yellow", "red"]],
        "ProceedToNextSignalManright": [("14-16", "12-14", "13-15"), (180, 180, 180), ["green", "yellow", "red"]],
        "StraightAhead": [("12-14", "14-16", "11-13", "13-15"), (180, -90, 0, -90), ["blue", "green"]],
        "StartEngines": [("12-14", "14-16", "11-13", "13-15"), (180, -90, -40, -40), ["yellow", "red", "red2", "blue", "green"]],
        "CutEngines": [("12-14", "14-16", "11-13", "13-15"), (180, -20, -90, -90), ["red", "red2", "blue"]],
        "SlowDown": [("12-14", "14-16", "11-13", "13-15"), (120, 180, 60, 0), ["yellow"]],
        "MoveBack": [("12-14", "14-16", "11-13", "13-15"), (120, 0, 60, 180), ["yellow", "red", "blue", "green"]],
        "Affirmative/AllClear": [("12-14", "14-16", "11-13", "13-15"), (180, -90, 90, 90), ["green"]],
        "Negative": [("12-14", "14-16", "11-13", "13-15"), (180, 180, 90, 90), ["red"]],
        #"Hover": [("12-14", "14-16", "11-13", "13-15"), (180, 180, 0, 0), []],
        #"HoldPosition/StandBy": [("12-14", "14-16", "11-13", "13-15"), (110, 110, 70, 70), ["green", "red"]],
        "DispatchAircraft": [("12-14", "14-16", "11-13", "13-15"), (180, -50, 90, 90), ["red", "blue", "green"]],
        "Connect/DisconnectPower": [("14-16", "11-13", "13-15"), (-60, -70, -150), ["blue"]],
        "EstablishCommunicationViaInterphone": [("12-14", "14-16", "11-13", "13-15"), (-160, -30, -30, -160), []],
        "Open/Close stairs": [("14-16", "11-13", "13-15"), (-30, -40, -60), ["red", "red2", "green"]],
        }
        
ANGLES_OF_INTEREST = set()
for reqs in JOINT_CONSTANTS.values():
    for joint_pair in reqs[0]:
        ANGLES_OF_INTEREST.add(joint_pair)

ANGLE_THRESHOLD = 30 # degrees

def turn_off_all_leds():
    for led in LEDS.values():
        led.off()

def make_decision(calc_angles: list):
    for sign, reqs in JOINT_CONSTANTS.items():
        if (check_requirements(reqs, calc_angles)):
            print("Sign predicted!")
            turn_off_all_leds()
            
            colors_to_light = reqs[2]
            for color in colors_to_light:
                if color in LEDS:
                    LEDS[color].on()
            return sign
    turn_off_all_leds()
    return "Waiting"

def check_requirements(req: list, calc_angles: list) -> bool:
    if isinstance(req[1], int):
        angles = [req[1]]
    else:
        angles = req[1]

    conds : bool = []
    for pair, angle in zip(req[0], angles):
        try:
            angle_diff = (calc_angles[pair] - angle + 180) % 360 - 180
            conds.append(abs(angle_diff) <= ANGLE_THRESHOLD)
            #print(f"ANGLES {pair}: {calc_angles[pair]}\n")
        except:
            print(f"Angle for {pair} not found")
            return False
    if all(conds):
        return True
    else:
        return False

def calculate_angle(next_landmark, landmark):
    diff_y = next_landmark.y - landmark.y
    diff_x = next_landmark.x - landmark.x
    angle = math.degrees(math.atan2(diff_y, diff_x))
    return angle



main_joint_idxs = [12, 14, 16, 18, 20, 22, 11, 13, 15, 17, 19, 21]

mp_pose = mp.solutions.pose
#DRAWING
mp_drawing = mp.solutions.drawing_utils
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
        # DRAWING
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=0, circle_radius=0), # Dot color/size
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)  # Line color/size
        )
        h, w, c = frame.shape
        landmark_list = results.pose_landmarks.landmark
        for idx, landmark in enumerate(landmark_list):
            # current_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            # joints numbers drawing -------------------
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            # --------------------------------------
            if idx in main_joint_idxs:
                if idx == 22 or idx == 21: continue

                next_landmark = landmark_list[idx+2]
                current_angles[f"{idx}-{idx+2}"] = calculate_angle(next_landmark, landmark)

                # hand joints
                if idx in [15, 16]:
                    next_landmark = landmark_list[idx+4]
                    current_angles[f"{idx}-{idx+4}"] = calculate_angle(next_landmark, landmark)
                    next_landmark = landmark_list[idx+6]
                    current_angles[f"{idx}-{idx+6}"] = calculate_angle(next_landmark, landmark)
    else:
        print("Person is lost")

    predicted_gesture = make_decision(current_angles)

    cv2.putText(frame, predicted_gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    #angles view----------------
    y_offset = 70
    for joint_pair, angle in current_angles.items():
        
        if joint_pair in ANGLES_OF_INTEREST:
            angle_text = f"Joint {joint_pair}: {int(angle)}" 
        
            cv2.putText(frame, angle_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 20
    # -------------------------
    cv2.imshow('Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
