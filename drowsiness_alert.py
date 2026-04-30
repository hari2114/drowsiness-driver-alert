"""
Drowsiness / Driver Alert System
==================================
Detects driver drowsiness in real time using Eye Aspect Ratio (EAR)
computed from facial landmarks via dlib.

An audio alert fires when eye closure exceeds the safety threshold.

Install:
    pip install opencv-python dlib scipy pygame imutils
    # dlib also requires cmake and a C++ compiler.
    # On Windows:  pip install cmake, then pip install dlib
    # On Ubuntu:   sudo apt-get install build-essential cmake

You also need:
    shape_predictor_68_face_landmarks.dat
    → Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    Extract and place in the project folder.

Usage:
    python drowsiness_alert.py
    Press 'q' to quit.
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import time
import os


# ── Constants ────────────────────────────────────────────────────────────────
PREDICTOR_PATH  = "shape_predictor_68_face_landmarks.dat"
ALERT_SOUND     = "alert.wav"          # place a .wav file here, or generate one
EAR_THRESHOLD   = 0.25                 # below this → eyes are closing
EAR_CONSEC_FRAMES = 20                 # frames below threshold before alert fires

# dlib landmark indices for left/right eye
(L_START, L_END) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_START, R_END) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio (EAR).
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def generate_alert_sound(path: str):
    """Generate a simple beep .wav if alert.wav does not exist."""
    import wave, struct, math
    if os.path.exists(path):
        return
    sample_rate = 44100
    duration = 1.0
    freq = 880
    amplitude = 32767
    n_samples = int(sample_rate * duration)
    with wave.open(path, "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(n_samples):
            t = i / sample_rate
            value = int(amplitude * math.sin(2 * math.pi * freq * t))
            wav.writeframes(struct.pack("<h", value))
    print(f"[INFO] Generated alert sound: {path}")


def init_audio(path: str):
    """Initialize pygame mixer and return the alert sound."""
    generate_alert_sound(path)
    pygame.mixer.init()
    return pygame.mixer.Sound(path)


def draw_eye_contour(frame, eye_pts):
    hull = cv2.convexHull(eye_pts)
    cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)


def overlay_status(frame, ear: float, counter: int, alert: bool):
    status_color = (0, 0, 200) if alert else (0, 200, 0)
    status_text  = "DROWSY! WAKE UP!" if alert else "Awake"

    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Status: {status_text}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    if alert:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]),
                      (0, 0, 200), 6)


def main():
    if not os.path.exists(PREDICTOR_PATH):
        print(f"[ERROR] Landmark predictor not found: '{PREDICTOR_PATH}'")
        print("  Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return

    print("[INFO] Loading dlib models...")
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    alert_sound = init_audio(ALERT_SOUND)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    frame_counter = 0
    alarm_on      = False
    print("[INFO] Drowsiness detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            shape = predictor(gray, face)
            shape_np = face_utils.shape_to_np(shape)

            left_eye  = shape_np[L_START:L_END]
            right_eye = shape_np[R_START:R_END]

            left_ear  = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            draw_eye_contour(frame, left_eye)
            draw_eye_contour(frame, right_eye)

            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= EAR_CONSEC_FRAMES:
                    alarm_on = True
                    if not pygame.mixer.get_busy():
                        alert_sound.play()
            else:
                frame_counter = 0
                if alarm_on:
                    alarm_on = False
                    pygame.mixer.stop()

            overlay_status(frame, ear, frame_counter, alarm_on)

        if not faces:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("[DONE] System stopped.")


if __name__ == "__main__":
    main()
