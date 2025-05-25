
def detect_gestures(results):
    gestures = []
    for hand_landmarks in results.multi_hand_landmarks:
        fingers = []

        # Thumb
        fingers.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)

        # Fingers (index to pinky)
        for tip in [8, 12, 16, 20]:
            fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)

        total_fingers = fingers.count(True)

        if total_fingers == 0:
            gestures.append("fist")
        elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            gestures.append("v_sign")
        elif fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
            gestures.append("three_fingers")
        elif total_fingers == 5:
            gestures.append("palm")

    return gestures
