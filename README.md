🖐️ AirDraw: Gesture-Based Virtual Whiteboard

An interactive computer vision-based drawing system that allows users to draw, erase, and control a virtual whiteboard using only hand gestures — no mouse, keyboard, or stylus required.

Built using OpenCV and MediaPipe, this project demonstrates real-time gesture recognition, UI interaction, and multi-slide presentation capabilities.

🚀 Features

1. ✋ Gesture-Based Interaction
2. ✍️ Draw using index finger
3. 🖐️ Hover to navigate UI
4. 🧽 Erase using multi-finger gesture

🔄 Translate (move objects)

🎨 Drawing Tools
1. Multiple colors and shapes (Line, Circle, Square)
2. Adjustable pen thickness
3. Adjustable eraser size
4. Undo / Redo support

🧾 Presentation Mode

1. Multi-slide support
2. Insert custom images as slide backgrounds
3. Dynamic image scaling
4. Slide navigation (Next / Previous / Delete)

🧠 Smart UI

1. Floating top-bar interface
2. Dwell-based clicking (hover to select)
3. Dark mode support

💾 Export System

1. Save slides as clean PNG images (no UI overlay)
2. Presentation-ready outputs

📊 Evaluation Module

1. FPS & latency benchmarking
2. Jitter (stability) measurement
3. Gesture accuracy testing

🏗️ System Architecture

Webcam Input
     ↓
MediaPipe Hand Tracking (21 landmarks)
     ↓
Gesture Detection (Vector + Cosine Similarity)
     ↓
Canvas Engine (State + Rendering)
     ↓
UI Overlay + Interaction
     ↓
Output Frame / Saved Slides

📐 Core Concepts

🟢 Hand Tracking

Uses 21 hand landmarks from MediaPipe

🔵 Gesture Recognition

Based on vector angles between fingers

Uses cosine similarity to detect:
a) Draw
b) Hover
c) Erase
d) Translate

🟡 Mathematical Foundations

I. Euclidean Distance → radius, scaling
II. Cosine Similarity → gesture classification
III. Standard Deviation → jitter analysis

🧪 Evaluation Metrics

Metric	Description

FPS	Measures smoothness

Latency	Time per frame (ms)

Jitter	Stability of hand tracking

Accuracy	Gesture detection correctness

⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/airdraw.git
cd airdraw
2. Install dependencies
pip install opencv-python mediapipe numpy

▶️ Usage

1. Run main application
python main.py

3. Run evaluation tool
python evaluate.py

🎮 Controls

Keyboard
Key	Action
n	Next slide
p	Previous slide
x	Delete slide
i	Insert image
[ / ]	Scale image
c	Change color
h	Change shape
s	Save slides
z	Undo
y	Redo
d	Toggle dark mode
a	Clear canvas
q	Quit
Gestures
Gesture	Action
Index finger	Draw
Open hand	Hover
3 fingers	Erase
Index + Pinky	Translate

📁 Project Structure
├── main.py              # Main application
├── evaluate.py          # Evaluation metrics tool
├── canvas.py            # Drawing + UI logic
├── hands.py             # Gesture detection logic
├── util.py              # Math utilities

🔥 Key Highlights

i. Real-time gesture-based interaction

ii. Custom UI built entirely using OpenCV

iii. Multi-slide virtual presentation system

iv. Non-destructive image scaling

v. Robust evaluation framework

🚧 Future Improvements

🤖 ML-based gesture classification

🎤 Voice command integration

🧠 Shape recognition (auto-detect shapes)

🌐 Web-based version (WebRTC)

🥽 AR/VR integration
