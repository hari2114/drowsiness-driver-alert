# 😴 Drowsiness / Driver Alert System

Real-time drowsiness detection using Eye Aspect Ratio (EAR) computed from 68 facial landmarks (dlib). Fires an audio alarm when eye closure exceeds a safety threshold — designed to prevent driver fatigue accidents.

---

## How It Works
1. Detects face in each frame using dlib's HOG detector
2. Predicts 68 facial landmarks (eyes, nose, mouth, jawline)
3. Computes **Eye Aspect Ratio (EAR)** for both eyes
4. If `EAR < 0.25` for **20+ consecutive frames** → triggers alarm

```
EAR = (|p2−p6| + |p3−p5|) / (2 × |p1−p4|)
```
When eyes are open, EAR ≈ 0.3+. When closed, EAR ≈ 0.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download landmark predictor
Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2  
Extract and place `shape_predictor_68_face_landmarks.dat` in the project folder.

### 3. Run
```bash
python drowsiness_alert.py
```
Press `q` to quit.

---

## Configuration (edit in script)
| Variable | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | 0.25 | Lower = less sensitive |
| `EAR_CONSEC_FRAMES` | 20 | Frames before alarm fires |

---

## Project Structure
```
drowsiness-driver-alert/
├── drowsiness_alert.py
├── shape_predictor_68_face_landmarks.dat   ← download separately
├── alert.wav                               ← auto-generated if missing
├── requirements.txt
└── README.md
```

---

## Tech Stack
| Library | Purpose |
|---|---|
| `dlib` | Face detection + landmarks |
| `OpenCV` | Webcam capture & display |
| `scipy` | Euclidean distance (EAR) |
| `pygame` | Audio alert |
| `imutils` | Face utils helpers |
