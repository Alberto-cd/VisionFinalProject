import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
from enum import Enum


# Colors - BGR
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (255, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
ORANGE = (24, 138, 254)

# Points of interest on the hand landmarks
THUMB_POINTS_INDEX = [1, 2, 4]
INDEX_POINTS_INDEX = [5, 6, 8]
MIDDLE_POINTS_INDEX = [9, 10, 12]
RING_POINTS_INDEX = [13, 14, 16]
PINKY_POINTS_INDEX = [17, 18, 20]

class HandState(Enum):
    UNKNOWN = 0
    ROCK = 1
    PAPER = 2
    SCISORS = 3
    START = 4
    END = 5


# def palm_centroid(coordinates_list):
#     coordinates = np.array(coordinates_list)
#     centroid = np.mean(coordinates, axis=0)
#     centroid = int(centroid[0]), int(centroid[1])
#     return centroid

def calculate_angle(p1, p2, p3):
    l1 = np.linalg.norm(p2 - p3)
    l2 = np.linalg.norm(p1 - p3)
    l3 = np.linalg.norm(p1 - p2)
    angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
    return angle

def extract_landmarks_given_list(hand_landmarks, width, height, index_list):
    return np.array([
                    [hand_landmarks.landmark[i].x * width, hand_landmarks.landmark[i].y * height]
                    for i in index_list
                ])

def extract_landmarks_for_each_finger(hand_landmarks, width, height):
    thumb = extract_landmarks_given_list(hand_landmarks, width, height, THUMB_POINTS_INDEX)
    index = extract_landmarks_given_list(hand_landmarks, width, height, INDEX_POINTS_INDEX)
    middle = extract_landmarks_given_list(hand_landmarks, width, height, MIDDLE_POINTS_INDEX)
    ring = extract_landmarks_given_list(hand_landmarks, width, height, RING_POINTS_INDEX)
    pinky = extract_landmarks_given_list(hand_landmarks, width, height, PINKY_POINTS_INDEX)
    return thumb, index, middle, ring, pinky

def classify_hand(hand_landmarks, width, height):
    # Extract landmarks for each finger
    thumb, index, middle, ring, pinky = extract_landmarks_for_each_finger(hand_landmarks, width, height)

    # Calculate angles for each finger
    thumb_angle = calculate_angle(*thumb)
    index_angle = calculate_angle(*index)
    middle_angle = calculate_angle(*middle)
    ring_angle = calculate_angle(*ring)
    pinky_angle = calculate_angle(*pinky)

    print(thumb_angle, index_angle, middle_angle, ring_angle, pinky_angle)
    # Hand classification based on the angles
    if thumb_angle > 150 and index_angle > 130 and middle_angle > 130 and ring_angle > 130 and pinky_angle > 130:
        text = "Papel detectado"
        color = GREEN
    elif thumb_angle < 156 and index_angle < 120 and middle_angle < 120 and ring_angle < 120 and pinky_angle < 150:
        text = "Piedra detectada"
        color = YELLOW
    elif index_angle > 130 and middle_angle > 130 and thumb_angle < 150 and ring_angle < 130 and pinky_angle < 130:
        text = "Tijera detectada"
        color = RED
    elif index_angle > 130 and middle_angle < 130 and ring_angle < 130 and pinky_angle > 130:
        text = "Termina juego"
        color = WHITE
    elif index_angle < 130 and middle_angle < 130 and ring_angle < 130 and pinky_angle > 130 and thumb_angle > 140:
        text = "Comienza juego"
        color = WHITE
    else:
        text = "Mano desconocida"
        color = WHITE
    return text, color

def initialize_kalman():
    # Create the Kalman filter object
    kf = cv2.KalmanFilter(4, 2)
    # Initialize the state of the Kalman filter
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0]], np.float32) # Measurement matrix np.array of shape (2, 4) and type np.float32
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32) # Transition matrix np.array of shape (4, 4) and type np.float32
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4 # Process noise covariance np.array of shape (4, 4) and type np.float32

    # measurement = np.array((2, 1), np.float32)
    # prediction = np.zeros((2, 1), np.float32)

    # crop_hist = None
    return kf

def main():
    # mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    max_num_hands = 2
    kalman_filters = [initialize_kalman() for _ in range(max_num_hands)]

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:


        count_frames = 0
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            # ### KALMAN - Inicio ###
            # # Convert the frame to HSV
            # img_hsv = cv2.cvtColor(input_frame, cv2.COLOR_BGR2HSV)
            
            # # Compute the back projection of the histogram
            # img_bproject = cv2.calcBackProject([img_hsv], [0], crop_hist, [0, 180], 1)
            
            # # Apply the mean shift algorithm to the back projection
            # ret, track_window = cv2.meanShift(img_bproject, track_window, term_crit)
            # x_,y_,w_,h_ = track_window
            # # Compute the center of the object
            # c_x = x_ + w_//2
            # c_y = y_ + h_//2
            
            # # Predict the position of the object
            # prediction = kf.predict()

            # # Update the measurement and correct the Kalman filter
            # measurement = np.array([[c_x], [c_y]], np.float32)
            # kf.correct(measurement)
            
            # # Draw the predicted position
            # cv2.circle(input_frame, (int(prediction[0][0]), int(prediction[1][0])), 5, (0, 0, 255), -1)
            # cv2.circle(input_frame, (int(c_x), int(c_y)), 5, (0, 255, 0), -1)
            
            # ### KALMAN - Final ###

            results = hands.process(frame_rgb)
            
            # if results.multi_hand_landmarks and count_frames % 60 == 0:
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Rectangle calculation
                    coords = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks.landmark]
                    pt1 = min(coords)[0]-10, min(coords, key= lambda c: c[1])[1]-10
                    pt2 = max(coords)[0]+10, max(coords, key= lambda c: c[1])[1]+10

                    text, color = classify_hand(hand_landmarks, width, height)

                    # Draw
                    cv2.putText(frame, text, (50 + i*width//2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                    cv2.rectangle(frame, pt1, pt2, color, 2)
                    # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Hand Detection', frame)  # Show the frame with hand detection

            count_frames += 1

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):  # Press 'q' to quit
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()