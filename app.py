import streamlit as st
import os

# Fix OpenCV issue
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io
import gdown
import matplotlib.pyplot as plt
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Space Debris Detection", layout="wide")

# ------------------ MODEL DOWNLOAD ------------------
file_id = "1kQTZ1ItERBwKRd1bbip8Cc-5BZfdDcsY"

if not os.path.exists("best.pt"):
    gdown.download(id=file_id, output="best.pt", quiet=False)

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
        boxes_list = []
        centers_draw = []

        for r in results:
            for box in r.boxes:
                count += 1
                conf_score = float(box.conf[0])
                total_conf += conf_score

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes_list.append((x1, y1, x2, y2))

                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Debris {conf_score:.2f}"
                cv2.putText(img, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Center point
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centers_draw.append((cx, cy))
                cv2.circle(img, (cx, cy), 4, (255, 0, 0), -1)

                # Large object warning
                area = (x2 - x1) * (y2 - y1)
                if area > 50000:
                    cv2.putText(img, "LARGE OBJECT!", (x1, y1-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # ------------------ DRAW COLLISION LINES ------------------
        for i in range(len(centers_draw)):
            for j in range(i+1, len(centers_draw)):
                dist = np.sqrt(
                    (centers_draw[i][0] - centers_draw[j][0])**2 +
                    (centers_draw[i][1] - centers_draw[j][1])**2
                )
                if dist < 100:
                    cv2.line(img, centers_draw[i], centers_draw[j], (0,0,255), 2)

        with col2:
            st.subheader("🛰️ Detection Result")
            st.image(img, channels="BGR")

        # ------------------ METRICS ------------------
        st.metric("Detected Objects", count)

        avg_conf = total_conf / count if count > 0 else 0
        st.write(f"🎯 Average Confidence: {avg_conf:.2f}")

        # ------------------ TOTAL AREA ------------------
        total_area = 0
        for (x1, y1, x2, y2) in boxes_list:
            total_area += (x2 - x1) * (y2 - y1)

        st.write(f"📦 Total Debris Area: {total_area}")

        # ------------------ RISK SYSTEM ------------------
        def get_risk(count):
            if count <= 2:
                return "Low 🟢"
            elif count <= 5:
                return "Medium 🟡"
            else:
                return "High 🔴"

        risk_level = get_risk(count)

        # ------------------ COLLISION PREDICTION ------------------
        def collision_risk_calc(boxes):
            if len(boxes) < 2:
                return "Low 🟢", 0

            centers = []
            for b in boxes:
                x1, y1, x2, y2 = b
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centers.append((cx, cy))

            min_dist = float('inf')

            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dist = np.sqrt(
                        (centers[i][0] - centers[j][0])**2 +
                        (centers[i][1] - centers[j][1])**2
                    )
                    min_dist = min(min_dist, dist)

            if min_dist < 50:
                return "High 🔴", min_dist
            elif min_dist < 100:
                return "Medium 🟡", min_dist
            else:
                return "Low 🟢", min_dist

        collision_level, min_distance = collision_risk_calc(boxes_list)

        # ------------------ DISPLAY ------------------
        st.subheader("🚨 Risk Analysis")
        st.write(f"⚠️ Debris Risk Level: {risk_level}")
        st.write(f"🚨 Collision Risk: {collision_level}")
        st.write(f"📏 Min Distance: {min_distance:.2f}")

        if collision_level == "High 🔴":
            st.error("⚠️ High Collision Risk!")
        elif collision_level == "Medium 🟡":
            st.warning("⚠️ Moderate Collision Risk")
        else:
            st.success("✅ Safe Zone")

        # ------------------ HISTORY ------------------
        st.session_state.history.append(count)

        st.subheader("📈 Detection Trend")
        st.line_chart(st.session_state.history)

        # ------------------ BAR GRAPH ------------------
        data = {
            "Category": ["Detected", "Safe"],
            "Count": [count, max(10 - count, 0)]
        }
        df = pd.DataFrame(data)
        st.subheader("📊 Detection Analytics")
        st.bar_chart(df.set_index("Category"))

        # ------------------ ADVANCED GRAPH ------------------
        st.subheader("📊 Advanced Analytics")
        labels = ["Debris Count", "Distance"]
        values = [count, min_distance]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        st.pyplot(fig)

        # ------------------ DOWNLOAD ------------------
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        st.download_button("📥 Download Result", buf.getvalue(), "result.png")

# ------------------ WEBCAM MODE ------------------
elif mode == "Webcam":

    st.subheader("🎥 Live Webcam Detection")

    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        prev_time = 0

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Webcam not working")
                break

            results = model(frame, conf=confidence)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time

            cv2.putText(frame, f"FPS: {int(fps)}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            frame_placeholder.image(frame, channels="BGR")

        cap.release()
        cv2.destroyAllWindows()

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("🚀 AI Space Debris Detection | YOLOv8 + Streamlit")
