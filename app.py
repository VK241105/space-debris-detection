import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io
import os
import urllib.request
import gdown

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Space Debris Detection", layout="wide")
import os
import gdown

file_id = "1kQTZ1ItERBwKRd1bbip8Cc-5BZfdDcsY"

# ALWAYS download fresh file (overwrite)
gdown.download(id=file_id, output="best.pt", quiet=False, fuzzy=True)

from ultralytics import YOLO
model = YOLO("best.pt")

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Controls")

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
mode = st.sidebar.radio("Select Mode", ["Image Upload", "Webcam"])

st.sidebar.markdown("---")
st.sidebar.info("🚀 AI-Based Space Debris Detection System")

# ------------------ TITLE ------------------
st.title("🚀 Space Debris Detection Dashboard")
st.markdown("Detect space debris using AI (YOLOv8 + OpenCV + Streamlit)")

# ------------------ SESSION HISTORY ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ IMAGE MODE ------------------
if mode == "Image Upload":

    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Image")
            st.image(img, channels="BGR")

        # ------------------ DETECTION ------------------
        results = model(img, conf=confidence)

        count = 0
        total_conf = 0

        for r in results:
            for box in r.boxes:
                count += 1
                conf_score = float(box.conf[0])
                total_conf += conf_score

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label with confidence
                label = f"Debris {conf_score:.2f}"
                cv2.putText(img, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Size detection (advanced feature)
                area = (x2 - x1) * (y2 - y1)
                if area > 50000:
                    cv2.putText(img, "LARGE OBJECT!", (x1, y1-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        with col2:
            st.subheader("🛰️ Detection Result")
            st.image(img, channels="BGR")

        # ------------------ METRICS ------------------
        st.metric("Detected Objects", count)

        # ------------------ RISK SYSTEM ------------------
        avg_conf = total_conf / count if count > 0 else 0

        if count == 0:
            st.success("🟢 Low Risk: No debris detected")
        elif count <= 3 and avg_conf < 0.6:
            st.warning("🟡 Medium Risk: Moderate debris")
        else:
            st.error("🔴 High Risk: Dangerous debris level!")

        # ------------------ HISTORY ------------------
        st.session_state.history.append(count)

        # ------------------ LINE GRAPH ------------------
        st.subheader("📈 Detection Trend")
        st.line_chart(st.session_state.history)

        # ------------------ BAR GRAPH ------------------
        data = {
            "Category": ["Detected", "Safe Zone"],
            "Count": [count, max(10 - count, 0)]
        }
        df = pd.DataFrame(data)
        st.subheader("📊 Detection Analytics")
        st.bar_chart(df.set_index("Category"))

        # ------------------ DOWNLOAD RESULT ------------------
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")

        st.download_button("📥 Download Result", buf.getvalue(), "result.png")

# ------------------ WEBCAM MODE ------------------
elif mode == "Webcam":

    st.subheader("🎥 Live Webcam Detection")

    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Webcam not working")
                break

            results = model(frame, conf=confidence)

            count = 0

            for r in results:
                for box in r.boxes:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            frame_placeholder.image(frame, channels="BGR")

        cap.release()
        cv2.destroyAllWindows()

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("🚀 AI Space Debris Detection | YOLOv8 + Streamlit")