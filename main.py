import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange() 

def initialize_camera():
    for i in range(3):  
        print(f"Trying camera index {i}")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Successfully opened camera {i}")
            cap.set(3, 640)  
            cap.set(4, 480) 
            return cap
    
    print("No camera found. Using dummy video.")
    cap = cv2.VideoCapture(0)
    return cap

cap = initialize_camera()

def calculate_distance(p1, p2):
    """Calculate distance between two points"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def percentage_to_db(percentage):
    """Convert percentage to logarithmic dB scale"""
    min_db = volRange[0]
    max_db = volRange[1] 
    
    if percentage == 0:
        return min_db
    elif percentage == 100:
        return max_db
    else:
        return min_db + ((max_db - min_db) * (percentage / 100))

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame. Retrying...")
            time.sleep(0.1)
            continue
            
        img = cv2.flip(img, 1)
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(imgRGB)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                thumb_tip = (int(hand_landmarks.landmark[4].x * img.shape[1]),
                            int(hand_landmarks.landmark[4].y * img.shape[0]))
                index_tip = (int(hand_landmarks.landmark[8].x * img.shape[1]),
                            int(hand_landmarks.landmark[8].y * img.shape[0]))
                
                cv2.circle(img, thumb_tip, 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, index_tip, 10, (255, 0, 0), cv2.FILLED)
                cv2.line(img, thumb_tip, index_tip, (255, 0, 0), 3)
                
                distance = calculate_distance(thumb_tip, index_tip)
                
                vol_percentage = distance / 3
                
                vol_percentage = max(0, min(vol_percentage, 100))
                
                vol = percentage_to_db(vol_percentage)
                
                volume.SetMasterVolumeLevel(vol, None)
                
                cv2.putText(img, f'Volume: {int(vol_percentage)}%', 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(img, f'Distance: {int(distance)}px', 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                bar_width = 300
                bar_height = 20
                bar_x = 10
                bar_y = 200
                bar_length = int(vol_percentage * bar_width / 100)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 0, 0), 2)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), (255, 0, 0), cv2.FILLED)
        
        cv2.imshow('Hand Volume Control', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
