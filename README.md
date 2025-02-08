import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def count_raised_fingers(hand_landmarks):
    """
    Counts the number of raised fingers in a hand.
    Args:
        hand_landmarks: MediaPipe hand landmarks.
    Returns:
        int: Number of raised fingers.
    """
    # Define the tips of the fingers
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_folded = []
    
    for i in range(1, 5):  # Ignore thumb (special case)
        # Check if tip is above the middle joint
        finger_folded.append(
            hand_landmarks.landmark[finger_tips[i]].y <
            hand_landmarks.landmark[finger_tips[i] - 2].y
        )
    
    # Thumb is checked differently
    thumb_folded = hand_landmarks.landmark[finger_tips[0]].x > \
                   hand_landmarks.landmark[finger_tips[0] - 1].x
                   
    return len([state for state in finger_folded if state]) + int(thumb_folded)

# Start capturing video
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip the frame for a selfie-view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Count raised fingers
                num_fingers = count_raised_fingers(hand_landmarks)
                cv2.putText(frame, f'Fingers: {num_fingers}', (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display the result
        cv2.imshow('Hand Gesture Recognition', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()
