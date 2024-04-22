# python 3.11.0
import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

cam = cv.VideoCapture(0)

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        print("Camera not available")
        continue

    # Flip the video frame horizontally (flip code 1 for x-axis)
    frame = cv.flip(frame, 1)

    # Convert frame to RGB
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Example of adjusting the blue channel - This is a placeholder, you may need to experiment
    frame[:, :, 2] = frame[:, :, 2] * 0.9  # Decrease the blue channel by 10%

    hands_detected = hands.process(frame)

    if hands_detected.multi_hand_landmarks:
        for hand_landmarks in hands_detected.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )

    # Convert RGB back to BGR for displaying
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    cv.imshow("Show Video", frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cam.release()
