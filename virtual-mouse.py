import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Detect only one hand for simplicity
mp_drawing = mp.solutions.drawing_utils

# Initialize PyAutoGUI
screen_width, screen_height = pyautogui.size()

# Smoothing for cursor movement
smoothening = 5  # Higher values make the cursor smoother but slower
ploc_x, ploc_y = 0, 0  # Previous cursor location
cloc_x, cloc_y = 0, 0  # Current cursor location

# Click sensitivity
click_threshold = 30  # Distance between thumb and index finger for click
double_click_threshold = 0.5  # Time in seconds for double-click
last_click_time = 0

# Frame dimensions (reduced for better performance)
frame_width, frame_height = 640, 480

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark coordinates
            landmarks = hand_landmarks.landmark

            # Index finger (landmark 8)
            index_x = int(landmarks[8].x * frame_width)
            index_y = int(landmarks[8].y * frame_height)

            # Thumb (landmark 4)
            thumb_x = int(landmarks[4].x * frame_width)
            thumb_y = int(landmarks[4].y * frame_height)

            # Draw circles on index finger and thumb
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 255), cv2.FILLED)

            # Calculate distance between index finger and thumb
            distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5

            # Map hand position to screen coordinates
            mapped_x = np.interp(index_x, (0, frame_width), (0, screen_width))
            mapped_y = np.interp(index_y, (0, frame_height), (0, screen_height))

            # Smooth cursor movement
            cloc_x = ploc_x + (mapped_x - ploc_x) / smoothening
            cloc_y = ploc_y + (mapped_y - ploc_y) / smoothening

            # Move mouse cursor
            pyautogui.moveTo(cloc_x, cloc_y)
            ploc_x, ploc_y = cloc_x, cloc_y

            # Click logic
            if distance < click_threshold:
                current_time = time.time()
                if current_time - last_click_time < double_click_threshold:
                    pyautogui.doubleClick()  # Double-click
                else:
                    pyautogui.click()  # Single-click
                last_click_time = current_time

    # Display frame
    cv2.imshow("Virtual Mouse", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()