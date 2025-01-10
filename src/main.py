import cv2
from picamera2 import Picamera2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import time
import tkinter as tk
from PIL import Image, ImageTk


# Colors - BGR
GREEN = (0, 255, 0)
BLUE = (255, 120, 0)
YELLOW = (0, 255, 255)
RED = (25, 50, 255)
WHITE = (255, 255, 255)
ORANGE = (24, 138, 254)

# Points of interest on the hand landmarks
THUMB_POINTS_INDEX = [1, 2, 4]
INDEX_POINTS_INDEX = [5, 6, 8]
MIDDLE_POINTS_INDEX = [9, 10, 12]
RING_POINTS_INDEX = [13, 14, 16]
PINKY_POINTS_INDEX = [17, 18, 20]

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

    # Hand classification based on the angles
    if thumb_angle > 150 and index_angle > 130 and middle_angle > 130 and ring_angle > 130 and pinky_angle > 130:
        text = "Papel"
        color = GREEN
    elif thumb_angle < 156 and index_angle < 165 and middle_angle < 175 and ring_angle < 180 and pinky_angle < 180:
        text = "Piedra"
        color = YELLOW
    elif index_angle > 130 and middle_angle > 130 and thumb_angle < 150 and ring_angle < 130 and pinky_angle < 170:
        text = "Tijera"
        color = ORANGE
    elif index_angle < 130 and middle_angle < 130 and ring_angle < 130 and pinky_angle > 130 and thumb_angle > 140:
        text = "Comienza juego"
        color = WHITE
    elif index_angle > 130 and middle_angle < 130 and ring_angle < 130 and pinky_angle > 130:
        text = "Termina juego"
        color = WHITE
    else:
        text = "Mano desconocida"
        color = WHITE
    return text, color

def get_rectangle(hand_landmarks, width:int, height:int, margin:int=0):
    # Rectangle calculation
    coords = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks.landmark]
    pt1 = min(coords)[0]-margin, min(coords, key= lambda c: c[1])[1]-margin
    pt2 = max(coords)[0]+margin, max(coords, key= lambda c: c[1])[1]+margin
    return pt1, pt2

def initialize_kalman_filter():
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

    return kf

def track_kalman(frame, kf:cv2.KalmanFilter, crop_hist, track_window, n):
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)
    # Convert the frame to HSV
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Compute the back projection of the histogram
    img_bproject = cv2.calcBackProject([img_hsv], [0], crop_hist, [0, 180], 1)
    
    # Apply the mean shift algorithm to the back projection
    ret, track_window = cv2.meanShift(img_bproject, track_window, term_crit)
    x_,y_,w_,h_ = track_window

    # Compute the center of the object
    c_x = x_ + w_//2
    c_y = y_ + h_//2
    
    # Predict the position of the object
    prediction = kf.predict()

    # Update the measurement and correct the Kalman filter
    measurement = np.array([[c_x], [c_y]], np.float32)
    kf.correct(measurement)

    return kf, crop_hist, track_window, frame, (int(c_x), int(c_y))

def initialize_window_kalman(frame, hand_landmarks, kf):
    height, width, _ = frame.shape

    pt1, pt2 = get_rectangle(hand_landmarks, width, height, -20)

    x, y, w, h = max(pt1[0], 0), max(pt1[1], 0), pt2[0]-pt1[0], pt2[1]-pt1[1]
    track_window = (x, y, w, h)

    # Compute the center of the object
    cx = x + w//2
    cy = y + h//2

    # Initialize the state of the Kalman filter
    kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

    # Initialize the covariance matrix
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    
    #Predict the position of the object
    prediction = kf.predict()
    
    # Update the measurement and correct the Kalman filter
    x_pred, y_pred = prediction[0], prediction[1]
    measurement = np.array([[x_pred], [y_pred]], np.float32)
    kf.correct(measurement)

    # Crop the object
    crop = frame[y:y + h, x:x + w].copy()

    if crop.size == 0:
        return kf, crop_hist, track_window, cx, cy

    # Convert the cropped object to HSV
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Compute a list that contains a mask (which should segment white colors) for every image.
    crop_mask = cv2.inRange(hsv_crop, (0, 0, 171), (255, 205, 255))

    # Compute the histogram of the cropped object (Reminder: Use only the Hue channel (0-180))
    crop_hist = cv2.calcHist([hsv_crop], [0], mask=crop_mask, histSize=[180], ranges=[0, 180])
    cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)

    return kf, crop_hist, track_window, cx, cy

def judge_game(p1_draw, p2_draw):
    if p1_draw == p2_draw:
        return "Empate", WHITE

    p1_valid = p1_draw in ["Papel", "Piedra", "Tijera"]
    p2_valid = p1_draw in ["Papel", "Piedra", "Tijera"]
    if not p1_valid and not p2_valid:
        return "Empate", WHITE
    
    if p1_valid:
        if p1_draw == "Piedra":
            if p2_draw == "Papel":
                return "Gana Jugador 2", RED
            if p2_draw == "Tijera":
                return "Gana Jugador 1", BLUE

        if p1_draw == "Papel":
            if p2_draw == "Tijera":
                return "Gana Jugador 2", RED
            if p2_draw == "Piedra":
                return "Gana Jugador 1", BLUE

        if p1_draw == "Tijera":
            if p2_draw == "Piedra":
                return "Gana Jugador 2", RED
            if p2_draw == "Papel":
                return "Gana Jugador 1", BLUE
    else:
        return "Gana Jugador 2", RED

def main():
    mp_hands = mp.solutions.hands

    picam = Picamera2()
    picam.preview_configuration.main.size=(960, 540)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    max_num_hands = 2
    kalman_filters = [initialize_kalman_filter() for _ in range(max_num_hands)]
    crop_hists = [None for _ in range(max_num_hands)]
    track_windows = [None for _ in range(max_num_hands)]
    last_measurements = [(0,0), (10000000, 100000000)]
    texts = [None for _ in range(max_num_hands)]
    colors = [None for _ in range(max_num_hands)]
    rectangles = [None for _ in range(max_num_hands)]

    game_started = False
    countdown_started = False
    countdown_time = 5  # 5 segundos para empezar juego
    last_detection_time = time.time()
    game_frame = None
    end_game_count = 0

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        count_frames = 0
        while True:
            frame = picam.capture_array()
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            if count_frames % 30 == 0:
                kalman_filters = [initialize_kalman_filter() for _ in range(max_num_hands)]
                crop_hists = [None for _ in range(max_num_hands)]
                track_windows = [None for _ in range(max_num_hands)]
                last_measurements = [(0,0), (10_000, 10_000)]
                texts = [None for _ in range(max_num_hands)]
                colors = [None for _ in range(max_num_hands)]
                rectangles = [None for _ in range(max_num_hands)]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for i, (hand_landmarks, kf, crop_hist, track_window) in enumerate(zip(results.multi_hand_landmarks, kalman_filters, crop_hists, track_windows)):
                        kalman_filters[i], crop_hists[i], track_windows[i], cx, cy =  initialize_window_kalman(frame, hand_landmarks, kf)

                        texts[i], colors[i] = classify_hand(hand_landmarks, width, height)
                        if texts[i] == "Comienza juego" and not countdown_started and not game_started and len(results.multi_hand_landmarks) == 2:
                            countdown_started = True
                            last_detection_time = time.time()
                        elif texts[i] == "Termina juego":
                            if not game_started and not countdown_started:
                                countdown_started = True
                                last_detection_time = time.time()
                                end_game_count += 1
                        elif end_game_count > 0:
                            countdown_started = False
                            end_game_count = 0

                        # Start the game
                        if game_started:
                            rectangles[i] = get_rectangle(hand_landmarks, width, height, 10)
                        # text, color = classify_hand(hand_landmarks, width, height)
                    if len(results.multi_hand_landmarks) == 2:
                        if ((last_measurements[0][0] - cx)**2 + (last_measurements[0][1] - cy)**2)**(1/2) < ((last_measurements[1][0] - cx)**2 + (last_measurements[1][1] - cy)**2)**(1/2):
                            kalman_filters[0], kalman_filters[1] = kalman_filters[1], kalman_filters[0]
                            crop_hists[0], crop_hists[1] = crop_hists[1], crop_hists[0]
                            track_windows[0], track_windows[1] = track_windows[1], track_windows[0]
                            last_measurements[0], last_measurements[1] = last_measurements[1], last_measurements[0]
                            texts[0], texts[1] = texts[1], texts[0]
                            colors[0], colors[1] = colors[1], colors[0]
                            rectangles[0], rectangles[1] = rectangles[1], rectangles[0]
            
            for i, (kf, crop_hist, track_window) in enumerate(zip(kalman_filters, crop_hists, track_windows)):
                if crop_hist is not None and track_window is not None:
                    kalman_filters[i], crop_hists[i], track_windows[i], frame, last_measurements[i] = track_kalman(frame, kf, crop_hist, track_window, i)

            if countdown_started:
                # Update countdown every second until it reaches 0
                elapsed_time = time.time() - last_detection_time
                remaining_time = countdown_time - int(elapsed_time)

                if remaining_time > 0:
                    if end_game_count == 0:
                        cv2.putText(frame, "Saca piedra", (width // 2 - 150, height // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame, "papel o tijera", (width // 2 - 150, height // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "Terminando juego", (width // 2 - 200, height // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.putText(frame, f"{remaining_time}", (width // 2, height // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
                else:
                    # Countdown finished, start the game and reset the flag
                    game_started = True
                    countdown_started = False
                    count_frames = -1 # en el siguiente frame se clasificarán las manos
            
            if game_frame is None: # or not game_started # pero esta segunda parte va implícita
                for i, crop_hist in enumerate(crop_hists):
                    if crop_hist is not None:
                        cv2.putText(frame, f"Jugador {i+1}", (50 + (i if last_measurements[0][0] < last_measurements[1][0] else 1-i)*width//2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE if i == 0 else RED, 2, cv2.LINE_AA)
                        cv2.circle(frame, (last_measurements[i][0], last_measurements[i][1]), 5, BLUE if i == 0 else RED, -1)
                if game_started and count_frames == 0:
                    for i, (text, color, rectangle) in enumerate(zip(texts, colors, rectangles)):
                        if rectangle is not None:
                            cv2.putText(frame, text, (50 + (i if last_measurements[0][0] < last_measurements[1][0] else 1-i)*width // 2, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                            cv2.rectangle(frame, rectangle[0], rectangle[1], color, 2)
                    result_game_text, result_game_color = judge_game(texts[0], texts[1])
                    cv2.putText(frame, result_game_text, (width // 2 - 100, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, result_game_color, 2, cv2.LINE_AA)
                    game_frame = frame

            if game_started:
                if (count_frames+2) % 60 == 0:
                    game_started = False
                    game_frame = None
                elif game_frame is not None:
                    frame = game_frame
            
            cv2.imshow('Hand Detection', frame)

            count_frames += 1

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or (end_game_count > 0 and countdown_started == False):  # Press 'q' to quit
                break

        cv2.destroyAllWindows()

def start_game():
    root.withdraw()  # Hide the start menu window
    try:
        main()  # Start the main game function
    except Exception:
        cv2.destroyAllWindows()
    root.quit()  # Close Tkinter interface after the game starts

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Piedra, Papel o Tijera")
    root.geometry("1000x800")
    root.configure(bg='#f0f0f0')

    # Title of the interface
    title_label = tk.Label(root, text="Juego: Piedra, Papel o Tijera", font=("Helvetica Neue", 24, 'bold'), fg='#3a3a3a', bg='#f0f0f0', pady=30)
    title_label.pack()

    # Start button
    start_button = tk.Button(root, text="Empezar", font=("Helvetica Neue", 18, 'bold'), bg='#4CAF50', fg='white', width=15, height=2, relief="flat", bd=2, command=start_game)
    start_button.pack(pady=20)
    start_button.config(activebackground="#45a049", activeforeground="white")

    # Image
    img = Image.open('img.jpg')
    img_tk = ImageTk.PhotoImage(img)
    label_img = tk.Label(root, image=img_tk)
    label_img.pack()

    # Run Tkinter interface
    root.mainloop()