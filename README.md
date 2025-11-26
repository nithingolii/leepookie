# Leepookie Meme Reactor (AI Gesture → Meme Player)

A real‑time gesture‑controlled meme generator using **MediaPipe Holistic**, a **machine‑learning gesture classifier**, and **OpenCV**.  

When you perform a gesture in front of your webcam, the app predicts the gesture and instantly plays the matching meme GIF.

Perfect for VS Code demos, TikTok/Reels videos, and coding fun with friends.

---
## ✨ Features
- Real‑time webcam tracking (face + hands)
- ML‑trained gesture classifier (Random Forest)
- Smooth, stable predictions using majority‑vote filtering
- Animated GIF playback
- Easy dataset capture and model retraining
- Fully customizable gesture → meme mapping

---
## 📦 Project Structure
```
MemeProject/
│
├── data_capture.py       # Capture gesture samples
├── train_model.py        # Train RandomForest model
├── build_dataset.py      # (Optional) Build dataset from .npy files
├── main.py               # Final real-time meme reactor
│
├── gesture_model.pkl     # Model generated after training
├── dataset.npz           # Combined dataset
│
├── dataset/              # Raw .npy per-sample files
│   ├── baby/
│   ├── dog/
│   ├── lebron/
│   ├── mj/
│   ├── rabbit/
│   └── shaq/
│
├── memes/ (optional)     # Or keep memes in project root
│   ├── niche_baby.jpg
│   ├── lebron_james.gif
│   ├── mj.gif
│   ├── rabbit.gif
│   ├── shaq.gif
│   └── dog.gif
```

---
## 🧪 Supported Gestures
| Gesture | Meaning | Meme Triggered |
|--------|----------|----------------|
| Hand on mouth | "Baby" expression | niche_baby.jpg |
| Hand on head | "Lebron frustrated" | lebron_james.gif |
| T‑pose with hands | Timeout gesture | shaq.gif |
| Pointing / holding | Rabbit meme | rabbit.gif |
| Crossed arms | MJ "Stop it" meme | mj.gif |
| No gesture (default) | Idle | dog.gif |

---
## 🛠 Installation
### 1. Install dependencies
```
pip install mediapipe opencv-python numpy scikit-learn joblib imageio pillow
```

---
## 🎥 Step 1: Capture Gesture Samples
Run:
```
python data_capture.py
```

Press these keys while performing each gesture:
```
0 → dog (default)
1 → baby (hand on mouth)
2 → lebron (hand on head)
3 → shaq (T pose)
4 → rabbit (pointing/holding)
5 → mj (crossed arms)
```
Each press saves one sample.

▶️ Recommended: **40+ samples per gesture** for good accuracy.

After collecting samples, press:
```
c → combine into dataset.npz
```

---
## 🤖 Step 2: Train the Model
Run:
```
python train_model.py
```
This will:
- load dataset.npz
- train a RandomForest classifier
- evaluate accuracy
- save `gesture_model.pkl`

---
## 🚀 Step 3: Run the Meme Reactor
```
python main.py
```
You will see two windows:
- **leepookie cam** → webcam view
- **leepookie reaction** → meme GIF based on prediction

Press `q` to quit.

---
## ⚙️ Optional: Rebuild Dataset Manually
If you added `.npy` files manually, run:
```
python build_dataset.py
```
This regenerates `dataset.npz` without recapturing.

---
## 🔧 Troubleshooting
### Model is confused / misdetecting
- Ensure 30–50 samples **per gesture**
- Vary lighting, distance, and angle while capturing
- Retrain using `train_model.py`

### GIFs not animating
`imageio` is required.
```
pip install imageio
```

### Webcam not opening
If using VS Code, run from a **local terminal**, not inside a remote environment.

---
## 📘 Tips for Better Accuracy
- Keep your face centered during capture
- Avoid harsh backlight
- Move slightly between each saved sample
- Capture at different distances
- Add more gestures easily by updating labels and retraining

---
## 🧩 Customize Your Own Gestures
Add a new folder under `dataset/`, capture `.npy` samples, rebuild dataset, retrain, and map it to any meme.

---
## 🧑‍💻 Credits
Created by **Nithin** — gesture-based meme reactor built with ML and MediaPipe.

Friends are free to modify, break, remix, and create meme chaos.

---
## ⭐ If you share this on GitHub
Consider adding project tags:
```
mediapipe, opencv, machine-learning, gestures, cv2, meme-generator, python-project
```

---
## 🎉 Enjoy creating chaos with gestures!

