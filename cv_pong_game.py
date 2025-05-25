import cv2
import mediapipe as mp
from gesture_utils import detect_gestures
import datetime

# Initialize video capture
cap = cv2.VideoCapture(0)

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Game variables
paddle_left_y = 250
paddle_right_y = 250
ball_x, ball_y = 320, 240
ball_dx, ball_dy = 5, 5
score_left, score_right = 0, 0
paused = False
game_over = False
MAX_SCORE = 5  # Set max score to end game


def reset_ball():
    global ball_x, ball_y, ball_dx, ball_dy
    ball_x, ball_y = 320, 240
    ball_dx = -ball_dx  # Reverse direction


# Game loop
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    gestures = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        gestures = detect_gestures(results)

        for idx, hand in enumerate(results.multi_hand_landmarks):
            cy = int(hand.landmark[mp_hands.HandLandmark.WRIST].y * h)
            if idx == 0:
                paddle_left_y = cy - 50
            elif idx == 1:
                paddle_right_y = cy - 50

    # Gesture controls
    if "fist" in gestures and not game_over:
        paused = not paused

    if "v_sign" in gestures:
        ball_dx *= 1.15
        ball_dy *= 1.15

    if "three_fingers" in gestures:
        score_left, score_right = 0, 0
        reset_ball()
        paused = False
        game_over = False

    if not paused and not game_over:
        ball_x += int(ball_dx)
        ball_y += int(ball_dy)

        # Ball bounce on top/bottom
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

    # Draw paddles and ball
    cv2.rectangle(frame, (20, paddle_left_y), (30, paddle_left_y + 100), (255, 0, 0), -1)
    cv2.rectangle(frame, (610, paddle_right_y), (620, paddle_right_y + 100), (0, 255, 0), -1)
    cv2.circle(frame, (ball_x, ball_y), 10, (0, 0, 255), -1)

    # Scores
    cv2.putText(frame, f"{score_left}", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, f"{score_right}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    if paused and not game_over:
        cv2.putText(frame, "Paused", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)

    if game_over:
        winner = "Left" if score_left >= MAX_SCORE else "Right"
        cv2.putText(frame, f"Game Over! {winner} Wins", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(frame, "Show Three Fingers to Restart", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("CV Pong", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
