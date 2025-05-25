import cv2
import mediapipe as mp
from gesture_utils import detect_gestures

# Initialize video capture
cap = cv2.VideoCapture(0)

# MediaPipe setups
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.8)

# Game variables
paddle_left_y = 250
paddle_right_y = 250
ball_x, ball_y = 320, 240
ball_dx, ball_dy = 5, 5
score_left, score_right = 0, 0
paused = False
game_over = False
MAX_SCORE = 5

# AI control flags
left_human = False
right_human = False

def reset_ball():
    global ball_x, ball_y, ball_dx, ball_dy
    ball_x, ball_y = 320, 240
    ball_dx = -ball_dx
    ball_dy = 5

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_results = face_detection.process(rgb)
    left_human, right_human = False, False
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            cx = bbox.xmin + bbox.width / 2
            if cx < 0.5:
                left_human = True
            else:
                right_human = True

    # Detect hand gestures
    results = hands.process(rgb)
    gestures = []

    if results.multi_hand_landmarks:
        gestures = detect_gestures(results)
        for hand in results.multi_hand_landmarks:
            wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
            cx = wrist.x
            cy = int(wrist.y * h)

            if cx < 0.5 and left_human:
                paddle_left_y = cy - 50
            elif cx >= 0.5 and right_human:
                paddle_right_y = cy - 50


    # Gesture controls
    if "fist" in gestures and not game_over:
        paused = not paused

    if "v_sign" in gestures:
        ball_dx *= 1.1
        ball_dy *= 1.1

    if "three_fingers" in gestures:
        score_left, score_right = 0, 0
        reset_ball()
        paused = False
        game_over = False

        # Ball movement
    if not paused and not game_over:
        ball_x += int(ball_dx)
        ball_y += int(ball_dy)

        # Wall bounce
        if ball_y <= 0 or ball_y >= 480:
            ball_dy = -ball_dy


        # Ball hits left paddle
        if ball_x <= 30:
            if paddle_left_y <= ball_y <= paddle_left_y + 100:
                score_left += 1
                ball_dx = -ball_dx
            else:
                score_right += 1
                reset_ball()

        # Ball hits right paddle
        elif ball_x >= 610:
            if paddle_right_y <= ball_y <= paddle_right_y + 100:
                score_right += 1
                ball_dx = -ball_dx
            else:
                score_left += 1
                reset_ball()

        # Game over check
        if score_left >= MAX_SCORE or score_right >= MAX_SCORE:
            paused = True
            game_over = True


    # AI paddle movement
    if not left_human and not paused:
        if paddle_left_y + 50 < ball_y:
            paddle_left_y += 5
        elif paddle_left_y + 50 > ball_y:
            paddle_left_y -= 5

    if not right_human and not paused:
        if paddle_right_y + 50 < ball_y:
            paddle_right_y += 5
        elif paddle_right_y + 50 > ball_y:
            paddle_right_y -= 5

    # Draw UI
    frame = cv2.line(frame, (320, 0), (320, 480), (100, 100, 100), 2)
    cv2.rectangle(frame, (20, paddle_left_y), (30, paddle_left_y + 100), (255, 0, 0), -1)
    cv2.rectangle(frame, (610, paddle_right_y), (620, paddle_right_y + 100), (0, 255, 0), -1)
    cv2.circle(frame, (ball_x, ball_y), 10, (0, 0, 255), -1)

    # Scores
    cv2.putText(frame, f"{score_left}", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, f"{score_right}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Status texts
    if paused and not game_over:
        cv2.putText(frame, "Paused", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)

    if game_over:
        winner = "Left" if score_left >= MAX_SCORE else "Right"
        cv2.putText(frame, f"Game Over! {winner} Wins", (130, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(frame, "Show Three Fingers to Restart", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if not left_human:
        cv2.putText(frame, "Left AI", (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,255), 2)
    if not right_human:
        cv2.putText(frame, "Right AI", (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,255), 2)

    cv2.imshow("Pong Game - Face + Gesture + AI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
