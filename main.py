import cv2
import mediapipe as mp
import os
import time

#Model path
model_path = "models/hand_landmarker.task"
if not os.path.exists(model_path):
    print("Hand model not found!")
    exit()

print("Hand model found!")


#Mediapipe setup
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options = BaseOptions(
        model_asset_path = model_path
    ),
    running_mode = VisionRunningMode.VIDEO,
    num_hands = 1
)

#Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access the camera!")
    exit()

print("Camera started!")
print("Show your hand to the camera.")
print("Press 'e' to exit.")

#Hand Landmarker
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()

        if not success:
            print("Could not read the camera frame!")
            break

        #BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        #frame to mediapipe image
        mp_image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = rgb_frame
        )
        timestamp_ms = int(time.time() * 1000)

        #Detect hands
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )
        if result.hand_landmarks:
            cv2.putText(
                frame,
                "Hand Detected",
                (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            #taking the first detected hand
            hand = result.hand_landmarks[0]

            #Draw Landmarks
            for landmark in hand:
                #normalized cordinated to pixel cordinates
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                #Draw a circle at the landmark position
                cv2.circle(
                    frame,
                    (x,y),
                    5,
                    (0,0,255),
                    -1
                )

        else:
            cv2.putText(
                frame,
                "No Hand Detected",
                (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        #Show webcam
        cv2.imshow("SkyWriting.ap - Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("e"):
            break

cap.release()
cv2.destroyAllWindows()
