# 🚀 Space Debris Detection System

An AI-powered web application that detects space debris from images and videos using YOLO (You Only Look Once) object detection model.

---

## 📌 Project Overview

Space debris is a growing threat to satellites and space missions. This project uses deep learning to automatically detect debris objects and analyze their potential risk.

---

## 🎯 Features

* 🔍 Image & video debris detection
* 🧠 YOLOv8 deep learning model
* 📊 Debris count & risk analysis
* 📈 Graph visualization
* 🎥 Upload support (image/video)
* 🌐 Interactive Streamlit UI
* ⚡ Fast & lightweight

---

## 🛠️ Tech Stack

* Python
* Streamlit
* YOLO (Ultralytics)
* OpenCV
* NumPy, Pandas
* Matplotlib

---

## 📂 Project Structure

```bash
space-debris-detection/
│
├── app.py
├── best.pt
├── requirements.txt
├── runtime.txt
├── dataset/ (optional)
└── README.md
```

---

## 📊 How It Works

1. Upload image or video
2. Model processes input
3. Detects debris using YOLO
4. Shows:

   * Bounding boxes
   * Confidence score
   * Total debris count
   * Risk level

---

## ⚠️ Risk Level Logic

| Count | Risk      |
| ----- | --------- |
| 0–2   | Low 🟢    |
| 3–5   | Medium 🟡 |
| 6+    | High 🔴   |

---

## 📥 Installation

```bash
git clone https://github.com/VK241105/space-debris-detection.git
cd space-debris-detection
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

Deployed using Streamlit Cloud:

1. Upload code to GitHub
2. Connect repo to Streamlit Cloud
3. Add requirements.txt & runtime.txt
4. Deploy 🚀

---

## 📁 Dataset

* Dataset sourced from Kaggle / custom images
* Annotated in YOLO format

---

## 🤖 Model Training

* YOLOv8 (Ultralytics)
* Trained on custom debris dataset
* Multiple epochs for accuracy

---

## 🔮 Future Scope

* 🛰️ Real-time debris tracking
* 📡 Satellite integration
* 🤖 Collision prediction AI
* 🌍 Live monitoring dashboard

---

## 👩‍💻 Author

**Vaishnavi Mane**
GitHub: https://github.com/VK241105

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
