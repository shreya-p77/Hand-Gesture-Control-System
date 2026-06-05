import cv2
import mediapipe as mp
import webbrowser
import subprocess
import os
import urllib.request

# --- Download model if needed ---
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand model... please wait")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
        model_path
    )
    print("Download done!")

# --- MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)

def fingers_up(landmarks, w, h):
    lm = [[int(l.x * w), int(l.y * h)] for l in landmarks]
    fingers = []
    fingers.append(1 if lm[4][0] < lm[3][0] else 0)
    for tip in [8, 12, 16, 20]:
        fingers.append(1 if lm[tip][1] < lm[tip - 2][1] else 0)
    return sum(fingers[1:]), fingers[0]

def close_all_browsers():
    for name in ['chrome', 'msedge', 'firefox', 'opera']:
        subprocess.run(['powershell', '-Command',
                        f'Stop-Process -Name {name} -Force -ErrorAction SilentlyContinue'],
                       capture_output=True)

cap = cv2.VideoCapture(0)

cv2.namedWindow("TD's Hand Gesture Control")
cv2.setWindowProperty("TD's Hand Gesture Control", cv2.WND_PROP_TOPMOST, 1)

gesture_hold = 0
current_gesture = ""
HOLD_FRAMES = 20
last_triggered = ""

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    cv2.putText(img, '1=Claude 2=ChatGPT 3=Gemini 4=LinkedIn Fist=CloseAll',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

            finger_count, thumb = fingers_up(hand, w, h)

            if finger_count == 0 and thumb == 0:
                gesture = "fist"
            else:
                gesture = str(finger_count)

            cv2.putText(img, f'Fingers: {finger_count} | Gesture: {gesture}',
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if gesture == current_gesture:
                gesture_hold += 1
            else:
                current_gesture = gesture
                gesture_hold = 0
                last_triggered = ""

            if gesture_hold == HOLD_FRAMES and last_triggered != gesture:
                last_triggered = gesture

                if gesture == "1":
                    webbrowser.open_new_tab('https://claude.ai')

                elif gesture == "2":
                    webbrowser.open_new_tab('https://chat.openai.com')

                elif gesture == "3":
                    webbrowser.open_new_tab('https://gemini.google.com')

                elif gesture == "4":
                    webbrowser.open_new_tab('https://www.linkedin.com')

                elif gesture == "fist":
                    close_all_browsers()

            labels = {
                "1": ("Opening Claude!", (255, 0, 255)),
                "2": ("Opening ChatGPT!", (0, 255, 0)),
                "3": ("Opening Gemini!", (0, 200, 255)),
                "4": ("Opening LinkedIn!", (255, 150, 0)),
                "fist": ("Closing All!", (0, 0, 255))
            }
            if gesture in labels and gesture_hold >= HOLD_FRAMES:
                text, color = labels[gesture]
                cv2.putText(img, text, (130, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    else:
        gesture_hold = 0
        current_gesture = ""
        last_triggered = ""

    cv2.imshow("TD's Hand Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()