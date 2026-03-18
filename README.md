# 🚀 Space Debris Detection System

An AI-powered web application that detects space debris from images and videos using YOLO (You Only Look Once) object detection model.

---

## 📌 Project Overview

Space debris poses a serious threat to satellites and space missions. This project uses deep learning to detect debris objects in space images and provides insights like risk level, object count, and visualization.

---

## 🎯 Features

* 🔍 Real-time debris detection (Image & Video)
* 🧠 YOLO-based deep learning model
* 📊 Detection count & risk analysis
* 📈 Graph visualization of results
* 🎥 Upload image/video support
* 🌐 Streamlit interactive UI dashboard
* ⚡ Fast and lightweight deployment

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

```
SpaceDebris Project/
│
├── app.py                # Main Streamlit application
├── best.pt               # Trained YOLO model
├── requirements.txt      # Dependencies
├── runtime.txt           # Python version (for deployment)
├── dataset/              # (Optional) Training dataset
└── README.md             # Project documentation
```

---

## 📊 How It Works

1. User uploads an image or video
2. YOLO model processes the input
3. Detects space debris objects
4. Displays:

   * Bounding boxes
   * Confidence score
   * Total debris count
   * Risk level (Low / Medium / High)

---

## ⚠️ Risk Level Logic

| Debris Count | Risk Level |
| ------------ | ---------- |
| 0 - 2        | Low 🟢     |
| 3 - 5        | Medium 🟡  |
| 6+           | High 🔴    |

---

## 📥 Installation (Local Setup)

```bash
git clone https://github.com/your-username/space-debris-detection.git
cd space-debris-detection

pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

This project is deployed using Streamlit Cloud.

Steps:

1. Upload code to GitHub
2. Connect repository to Streamlit Cloud
3. Add `requirements.txt` and `runtime.txt`
4. Deploy 🚀

---

## 📁 Dataset

* Dataset used for training can be taken from Kaggle or custom collected images.
* Images are labeled in YOLO format.

---

## 🤖 Model Training

* YOLOv8 model used from Ultralytics
* Custom dataset trained for debris detection
* Trained for multiple epochs to improve accuracy

---

## 📸 Output Example

* Bounding boxes around debris
* Confidence scores
* Risk analysis graph

---

## 🔮 Future Enhancements

* 🔴 Real-time satellite tracking
* 🛰️ Live space debris monitoring
* 📡 API integration with space agencies
* 🤖 AI-based prediction of collision risk

---

## 👩‍💻 Author

Vaishnavi Mane
BTech CSE (AI & ML)

---

## ⭐ Acknowledgment

* Ultralytics YOLO
* Open-source datasets
* Streamlit for UI

---

## 📌 Note

This project is for educational and research purposes.

---

⭐ If you like this project, give it a star on GitHub!

