🖐️ AirDraw: Gesture-Based Virtual Whiteboard

An interactive computer vision-based drawing system that allows users to draw, erase, and control a virtual whiteboard using only hand gestures — no mouse, keyboard, or stylus required.

Built using OpenCV and MediaPipe, this project demonstrates real-time gesture recognition, UI interaction, and multi-slide presentation capabilities.

🚀 Features
✋ Gesture-Based Interaction
✍️ Draw using index finger
🖐️ Hover to navigate UI
🧽 Erase using multi-finger gesture

🔄 Translate (move objects)
🎨 Drawing Tools
Multiple colors and shapes (Line, Circle, Square)
Adjustable pen thickness
Adjustable eraser size
Undo / Redo support

🧾 Presentation Mode
Multi-slide support
Insert custom images as slide backgrounds
Dynamic image scaling
Slide navigation (Next / Previous / Delete)

🧠 Smart UI
Floating top-bar interface
Dwell-based clicking (hover to select)
Dark mode support

💾 Export System
Save slides as clean PNG images (no UI overlay)
Presentation-ready outputs

📊 Evaluation Module
FPS & latency benchmarking
Jitter (stability) measurement
Gesture accuracy testing

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
Draw
Hover
Erase
Translate

🟡 Mathematical Foundations
Euclidean Distance → radius, scaling
Cosine Similarity → gesture classification
Standard Deviation → jitter analysis

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
Run main application
python main.py
Run evaluation tool
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
Real-time gesture-based interaction
Custom UI built entirely using OpenCV
Multi-slide virtual presentation system
Non-destructive image scaling
Robust evaluation framework

🚧 Future Improvements

🤖 ML-based gesture classification
🎤 Voice command integration
🧠 Shape recognition (auto-detect shapes)
🌐 Web-based version (WebRTC)
🥽 AR/VR integration
