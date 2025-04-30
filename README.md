
---

# 🎭 Real-Time Emotion Recognition from Webcam

This project uses computer vision and deep learning to detect human faces and recognize emotions in real time through a webcam.

## 🚀 Features

- Real-time face detection using OpenCV
- Emotion recognition using a trained PyTorch model
- Easy to train on your own dataset
- Lightweight and runs smoothly on most systems

## 🧠 Emotion Classes

The model classifies emotions into the following categories:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📷 How to Use

### Run Emotion Detection:
```bash
python detect_emotion.py
```

The webcam will open, and the system will detect your face and display the predicted emotion.



Make sure your dataset is structured and labeled appropriately.

## 🧾 Requirements

- Python 3.7+
- PyTorch
- OpenCV
- NumPy

All dependencies can be installed via `requirements.txt`.

## 📁 Project Structure

```
emotion-recognition/
├── emotion-detection_train_model.ipynb  # Training script
├── detect_emotion.py      # Real-time emotion detection
├── models              # Trained models (optional, add download link)
├── requirements.txt
├── examples/
│   └── demo.jpg           # Sample image showing detection
└── README.md
```

## 📂 Dataset

The model was trained on the [FER2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). You can also use your own dataset following a similar format.

## 🖼️ Demo

![Emotion Detection Demo](examples/demo.jpg)


