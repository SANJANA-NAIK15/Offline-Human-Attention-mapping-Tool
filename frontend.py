import streamlit as st
import cv2
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Attention Monitor", layout="wide")

# -----------------------------
# PAGE TITLE
# -----------------------------

st.title("🧠 AI Attention Monitor")
st.subheader("Real-Time Focus Detection System")

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("Dashboard Controls")

start = st.sidebar.button("Start Monitoring")
stop = st.sidebar.button("Stop Monitoring")

# -----------------------------
# METRICS
# -----------------------------

col1, col2, col3 = st.columns(3)

focus_placeholder = col1.metric("Focus Rate", "0 %")
time_placeholder = col2.metric("Session Time", "0 sec")
score_placeholder = col3.metric("Productivity Score", "0 /100")

# -----------------------------
# VIDEO FRAME
# -----------------------------

video_placeholder = st.empty()

# -----------------------------
# GRAPH AREA
# -----------------------------

st.subheader("Focus Trend")

chart_placeholder = st.empty()

# -----------------------------
# FACE DETECTOR
# -----------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# START CAMERA
# -----------------------------

if start:

    cap = cv2.VideoCapture(0)

    start_time = time.time()
    attentive_time = 0

    focus_data = []

    while True:

        ret, frame = cap.read()

        if not ret:
            st.error("Camera not detected")
            break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        attentive = False

        if len(faces) > 0:

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

            cv2.putText(
                frame,
                "Not Attentive",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

        total_time = time.time() - start_time

        if attentive:
            attentive_time += 0.05

        focus_rate = (attentive_time / total_time) * 100 if total_time > 0 else 0

        productivity_score = min(int(focus_rate + 20),100)

        focus_data.append(focus_rate)

        # Update Metrics

        focus_placeholder.metric("Focus Rate", f"{focus_rate:.2f} %")
        time_placeholder.metric("Session Time", f"{int(total_time)} sec")
        score_placeholder.metric("Productivity Score", f"{productivity_score} /100")

        # Show Camera

        video_placeholder.image(frame, channels="BGR")

        # Update Graph

        fig, ax = plt.subplots()
        ax.plot(focus_data)
        ax.set_title("Focus Rate Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Focus %")

        chart_placeholder.pyplot(fig)

        if stop:
            break

    cap.release()