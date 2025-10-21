# 😊 Emotion Detection using OpenCV

## 📖 Overview
**Emotion Detection using OpenCV** is a computer vision-based project that detects human emotions in real-time using facial expressions captured through a webcam or image.  
The system uses **OpenCV** for image processing and **Machine Learning** models (like Haar Cascade and CNN) to recognize emotions such as **happy, sad, angry, surprised, neutral,** and more.

This project demonstrates how AI can understand human emotions and can be integrated into applications such as smart classrooms, mental health analysis, human-computer interaction, and intelligent surveillance systems.

---

## 🚀 Features
- 🎥 Real-time emotion detection using webcam.
- 🧠 Facial expression recognition using pre-trained models.
- 📷 Face detection with OpenCV’s Haar Cascade Classifier.
- 📊 Displays the predicted emotion on the video frame.
- 💾 Easy to extend with your own datasets and models.

---

## 🧰 Tech Stack
- **Programming Language:** Python  
- **Libraries Used:** OpenCV, TensorFlow / Keras, NumPy, Matplotlib  
- **Model Used:** CNN (Convolutional Neural Network) or Haar Cascade Classifier  
- **Environment:** Jupyter Notebook / VS Code / PyCharm  

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/emotion-detection-opencv.git
cd emotion-detection-opencv
```
2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the application
python emotion_detection.py

4️⃣ (Optional) Run on Jupyter Notebook

If your project uses a notebook:

jupyter notebook emotion_detection.ipynb

📁 Folder Structure
📂 emotion-detection-opencv/
 ┣ 📁 dataset/              # Emotion image dataset (e.g., FER2013)
 ┣ 📁 models/               # Trained models (if saved)
 ┣ 📁 haarcascades/         # Haar cascade XML files for face detection
 ┣ 📄 emotion_detection.py  # Main program
 ┣ 📄 requirements.txt      # List of dependencies
 ┣ 📄 README.md             # Project documentation
 ┗ 📄 LICENSE               # License (optional)

🧠 How It Works
1.The system captures a video feed or image using the webcam.
2.OpenCV detects the face region using a Haar Cascade classifier.
3.The cropped face is passed into the CNN model.
4.The model predicts the corresponding emotion (e.g., happy, sad, angry, etc.).
5.The detected emotion is displayed on the screen with bounding boxes.

🧩 Example Emotions Detected
😀 Happy
😢 Sad
😠 Angry
😮 Surprised
😐 Neutral
😨 Fear
🤢 Disgust
