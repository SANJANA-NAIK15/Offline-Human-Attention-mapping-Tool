import time
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pynput import keyboard, mouse
import win32gui

# -----------------------------
# VARIABLES
# -----------------------------

log = []
last_activity_time = time.time()
idle_threshold = 10
last_window = None
last_mouse_log = 0

attentive_time = 0
total_time = 0
start_time = time.time()

distraction_start = None
distraction_time = 0

# -----------------------------
# FACE DETECTOR
# -----------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# -----------------------------
# LOGGER
# -----------------------------

def log_event(event):
    global last_activity_time
    timestamp = datetime.now().strftime("%H:%M:%S")
    log.append([timestamp, event])
    last_activity_time = time.time()
    print(timestamp, event)

# -----------------------------
# KEYBOARD
# -----------------------------

def on_press(key):
    log_event("Keyboard Activity")

# -----------------------------
# MOUSE
# -----------------------------

def on_move(x, y):
    global last_mouse_log
    now = time.time()

    if now - last_mouse_log > 1:
        log_event("Mouse Movement")
        last_mouse_log = now

def on_click(x, y, button, pressed):
    if pressed:
        log_event("Mouse Click")

# -----------------------------
# START LISTENERS
# -----------------------------

keyboard_listener = keyboard.Listener(on_press=on_press)
mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)

keyboard_listener.start()
mouse_listener.start()

print("Tracking started... Press ESC to stop")

# -----------------------------
# MAIN LOOP
# -----------------------------

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        attentive = False

        # -----------------------------
        # FACE DETECTION
        # -----------------------------

        if len(faces) > 0:

            distraction_start = None
            attentive = True

            for (x, y, w, h) in faces:

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

                cv2.putText(
                    frame,
                    "Attentive",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

        else:

            attentive = False

            if distraction_start is None:
                distraction_start = time.time()

            distraction_time = time.time() - distraction_start

            cv2.putText(
                frame,
                f"Not Attentive {int(distraction_time)}s",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

        # -----------------------------
        # TIME + FOCUS RATE
        # -----------------------------

        total_time = time.time() - start_time

        if attentive:
            attentive_time += 0.05

        focus_rate = (attentive_time / total_time) * 100 if total_time > 0 else 0

        cv2.putText(
            frame,
            f"Focus Rate: {focus_rate:.2f}%",
            (30,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Time: {int(total_time)} sec",
            (30,110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        # -----------------------------
        # PRODUCTIVITY SCORE
        # -----------------------------

        keyboard_score = 20
        mouse_score = 20
        face_score = min(focus_rate,60)

        productivity_score = int(face_score + keyboard_score + mouse_score)

        cv2.putText(
            frame,
            f"Productivity Score: {productivity_score}/100",
            (30,140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        # -----------------------------
        # FOCUS ALERT
        # -----------------------------

        if focus_rate < 40:

            cv2.putText(
                frame,
                "Low Focus!",
                (350,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

        elif focus_rate > 80:

            cv2.putText(
                frame,
                "Great Focus!",
                (350,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                3
            )

        cv2.imshow("AI Attention Monitor", frame)

        # -----------------------------
        # WINDOW SWITCH DETECTION
        # -----------------------------

        window = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(window)

        if title and title != last_window:
            log_event("Window Switch -> " + title)
            last_window = title

        # -----------------------------
        # IDLE DETECTION
        # -----------------------------

        current_time = time.time()
        idle_time = current_time - last_activity_time

        if idle_time > idle_threshold:
            log_event("Idle")
            last_activity_time = current_time

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.05)

# -----------------------------
# STOP PROGRAM
# -----------------------------

except KeyboardInterrupt:
    print("\nStopping tracking...")

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# SAVE LOG
# -----------------------------

df = pd.DataFrame(log, columns=["Time", "Event"])
df.to_csv("interaction_log.csv", index=False)

print("Log saved as interaction_log.csv")

# -----------------------------
# ATTENTION HEATMAP
# -----------------------------

state_map = {
    "Keyboard Activity": 3,
    "Mouse Click": 3,
    "Mouse Movement": 2,
    "Idle": 1
}

states = []

for event in df["Event"]:
    if "Window Switch" in event:
        states.append(2)
    else:
        states.append(state_map.get(event,2))

size = int(np.sqrt(len(states))) + 1
padded = states + [1]*(size*size - len(states))
heatmap = np.array(padded).reshape(size,size)

plt.figure(figsize=(10,5))
plt.imshow(heatmap, cmap="RdYlGn", aspect="auto")

plt.title("User Attention Heatmap")
plt.colorbar(label="Attention Level")

plt.xticks([])
plt.yticks([])

plt.show()