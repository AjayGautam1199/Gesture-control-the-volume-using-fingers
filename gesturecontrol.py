import cv2
import mediapipe as mp
import time
import math
import numpy as np
import ctypes

# Initialize MediaPipe hands and drawing utilities
mphand = mp.solutions.hands
hands = mphand.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Set up webcam capture
cap = cv2.VideoCapture(0)

# Function to set Windows volume, scaled appropriately
def set_volume(vol):
    # Scale volume to fit within 0 to 100 range
    scaled_volume = max(0, min(100, vol))
    # Set Windows volume using ctypes
    volume = int(ctypes.c_float(scaled_volume / 100 * 65535).value)
    ctypes.windll.WINMM.waveOutSetVolume(0, volume | (volume << 16))

# Convert hand distance to volume level
def volume_level(length, min_len=50, max_len=300, min_vol=0, max_vol=100):
    # Ensure the volume is within the range specified
    return np.interp(length, [min_len, max_len], [min_vol, max_vol])

# Initialize timing and volume variables
prev_time = 0
vol_per = 0  # Default volume percentage initialization
lock_time = None  # Time when volume lock was activated
locked = False  # Flag to indicate if volume is locked

while True:
    success, img = cap.read()
    if not success:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    lm_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
            mp_draw.draw_landmarks(img, hand_landmarks, mphand.HAND_CONNECTIONS)

    if lm_list:
        # Calculate distance between thumb tip and index tip
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        length = math.hypot(x2 - x1, y2 - y1)
        vol_per = volume_level(length)
        
        # Check if volume is locked and unlock after 3 seconds
        if locked and time.time() - lock_time > 3:
            locked = False

        # Adjust volume if not locked
        if not locked:
            set_volume(vol_per)

            # Check if volume should be locked
            if vol_per == 0 or vol_per == 100:
                locked = True
                lock_time = time.time()

    # Display volume level and FPS
    cv2.putText(img, f"Vol: {int(vol_per)}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
