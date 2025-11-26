# MemeProject — dataset capture, training, and main app

# File: data_capture.py

"""
Capture MediaPipe landmarks for each gesture and save them to disk.
Usage:
  python data_capture.py
Keys:
  0 -> dog (default)
  1 -> baby (hand on mouth)
  2 -> lebron (hand on head)
  3 -> shaq (T pose)
  4 -> rabbit (pointing/holding)
  5 -> mj (crossed)
  c -> combine saved .npy files into dataset.npz (optional)
  q -> quit

It will save per-sample .npy files under ./dataset/<label>/sample_XXXX.npy

Make sure requirements are installed:
  pip install mediapipe opencv-python numpy

"""
import os
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

LABELS = {
    ord('0'): 'dog',
    ord('1'): 'baby',
    ord('2'): 'lebron',
    ord('3'): 'shaq',
    ord('4'): 'rabbit',
    ord('5'): 'mj'
}

OUTDIR = 'dataset'
CAM_ID = 0

mp_holistic = mp.solutions.holistic

os.makedirs(OUTDIR, exist_ok=True)
for lbl in set(LABELS.values()):
    os.makedirs(os.path.join(OUTDIR, lbl), exist_ok=True)


def extract_landmark_vector(results):
    # Collect landmarks from face (468), left hand (21), right hand (21)
    # For missing parts we fill zeros for consistent size.
    vec = []
    # face
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0.0] * 468 * 3)
    # left hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0.0] * 21 * 3)
    # right hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0.0] * 21 * 3)
    return np.array(vec, dtype=np.float32)


def main():
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print('ERROR: could not open camera')
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        sample_counters = {lbl: len(os.listdir(os.path.join(OUTDIR, lbl))) for lbl in set(LABELS.values())}
        print('Press keys 0-5 to capture samples. c to combine into dataset.npz. q to quit')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)

            # draw minimal overlay
            h, w = img.shape[:2]
            if results.face_landmarks:
                # draw rect
                xs = [lm.x * w for lm in results.face_landmarks.landmark]
                ys = [lm.y * h for lm in results.face_landmarks.landmark]
                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)

            cv2.putText(img, 'Press 0-5 to capture, c combine, q quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow('capture', img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in LABELS:
                label = LABELS[key]
                vec = extract_landmark_vector(results)
                # file name time-based
                fname = f"sample_{sample_counters[label]:04d}.npy"
                np.save(os.path.join(OUTDIR, label, fname), vec)
                sample_counters[label] += 1
                print(f'Saved {label}/{fname}')
            if key == ord('c'):
                # combine into dataset.npz for quick training
                X, y = [], []
                label_to_idx = {lbl: idx for idx, lbl in enumerate(sorted(set(LABELS.values())))}
                for lbl in label_to_idx:
                    folder = os.path.join(OUTDIR, lbl)
                    for f in os.listdir(folder):
                        if f.endswith('.npy'):
                            X.append(np.load(os.path.join(folder, f)))
                            y.append(label_to_idx[lbl])
                X = np.stack(X, axis=0)
                y = np.array(y)
                np.savez('dataset.npz', X=X, y=y, labels=list(label_to_idx.keys()))
                print('Saved dataset.npz with', X.shape[0], 'samples')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
