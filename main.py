
# File: main.py

"""
Real-time meme reactor using the trained gesture model.
Usage:
  python main.py
Requirements:
 pip install mediapipe opencv-python imageio pillow numpy joblib

Place the trained gesture_model.pkl (created by train_model.py) alongside the script.

"""
import cv2
import mediapipe as mp
import numpy as np
import imageio
import os
import joblib
from collections import deque, Counter

MODEL_FILE = 'gesture_model.pkl'
CAM_ID = 0
WINDOW_CAM = 'leepookie cam'
WINDOW_MEME = 'leepookie reaction'
MEME_DISPLAY_WIDTH = 560
SMOOTH_WINDOW = 6  # frames for majority vote smoothing

# mapping label index to meme key will be loaded from model
memefile_names = {
    'baby': 'niche_baby.jpg',
    'lebron': 'lebron_james.gif',
    'mj': 'mj.gif',
    'rabbit': 'rabbit.gif',
    'shaq': 'shaq.gif',
    'dog': 'dog.gif'
}

# load model
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError('gesture model not found. Run train_model.py first')
obj = joblib.load(MODEL_FILE)
model = obj['model']
labels = obj.get('labels', None)
if labels is None:
    # assume alphabetical fallback
    labels = sorted(list(memefile_names.keys()))
label_idx_to_name = {i: labels[i] for i in range(len(labels))}

# load meme frames

def load_media(path):
    if not os.path.exists(path):
        img = np.full((360, 640, 3), 200, dtype=np.uint8)
        cv2.putText(img, 'Missing '+os.path.basename(path), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        return [img]
    try:
        reader = imageio.get_reader(path)
        frames = []
        for im in reader:
            frames.append(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
        if len(frames)==0:
            return [cv2.imread(path)]
        return frames
    except Exception:
        img = cv2.imread(path)
        return [img]

memes_frames = {k: load_media(v) for k, v in memefile_names.items()}
for k in list(memes_frames.keys()):
    # resize
    resized = []
    for f in memes_frames[k]:
        h, w = f.shape[:2]
        scale = MEME_DISPLAY_WIDTH / float(w)
        resized.append(cv2.resize(f, (MEME_DISPLAY_WIDTH, int(h*scale))))
    memes_frames[k] = resized

# mediapipe util
mp_holistic = mp.solutions.holistic


def extract_landmark_vector(results):
    vec = []
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0.0] * 468 * 3)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0.0] * 21 * 3)
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

    frame_counters = {k: 0 for k in memes_frames.keys()}
    vote_buffer = deque(maxlen=SMOOTH_WINDOW)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)

            # minimal overlay
            h, w = img.shape[:2]
            if results.face_landmarks:
                xs = [lm.x * w for lm in results.face_landmarks.landmark]
                ys = [lm.y * h for lm in results.face_landmarks.landmark]
                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)

            # predict gesture
            vec = extract_landmark_vector(results).reshape(1, -1)
            pred_idx = model.predict(vec)[0]
            gesture = label_idx_to_name.get(int(pred_idx), 'dog')

            # smoothing
            vote_buffer.append(gesture)
            majority = Counter(vote_buffer).most_common(1)[0][0]

            # animate chosen
            frames = memes_frames.get(majority, memes_frames['dog'])
            idx = frame_counters[majority] % len(frames)
            meme_frame = frames[idx]
            frame_counters[majority] += 1

            meme_show = meme_frame.copy()
            cv2.putText(meme_show, majority.upper(), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2)

            cv2.imshow(WINDOW_CAM, img)
            cv2.imshow(WINDOW_MEME, meme_show)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

