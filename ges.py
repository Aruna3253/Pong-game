def detect_gestures(results):
    gestures = []

    # Prevent error if no hands are detected
    if not results.multi_hand_landmarks:
        return gestures

    for hand_landmarks in results.multi_hand_landmarks:
        # Dummy logic — replace with your actual finger detection
        # For now, let's just pretend it returns "fist" if hand is present
        gestures.append("fist")

    return gestures
