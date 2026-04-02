# Gesture-Controlled Smartboard (CV Canvas)

## 📖 Project Overview
This project is an advanced, real-time Computer Vision application that transforms standard hardware (a webcam) into an interactive digital smartboard. By leveraging precise hand-tracking algorithms, users can draw, erase, manipulate images, create shapes, and navigate through a multi-slide presentation entirely through air gestures. 

It eliminates the requirement for physical hardware like an interactive smartboard, mouse or stylus, providing a natural and intuitive interface tailored for classroom instruction, remote teaching, and interactive presentations.

## ✨ Core Features
- **High-Precision Air Drawing**: Uses machine learning hand landmarks to track the index finger for drawing, incorporating anti-tremor stabilization logic to produce smooth ink.
- **Dynamic Gesture Recognition**: Intuitively switches between modes (e.g., drawing, moving, erasing, saving) based on real-time hand configurations and distance calculations (e.g., pinching vs. open hand).
- **Multi-Slide Architecture**: Allows the creation of endless slides, seamless navigation (`n` / `p`), and persistent context preservation per slide.
- **Robust History Engine (Undo/Redo)**: Implements state management stacks to allow users to instantly revert or reapply their interactions without losing work.
- **Modern Glassmorphic UI**: Provides a modern, aesthetically pleasing visual interface with an on-screen Heads-Up Display (HUD) feedback overlay.

## 🛠️ System Architecture / Tech Stack
- **Python**: Core backend programming language.
- **OpenCV (`cv2`)**: Handles webcam video capture, frame buffer manipulation, drawing elements on the screen, and rendering the UI Canvas.
- **Google MediaPipe**: Machine learning framework used for high-fidelity, real-time 3D hand tracking and landmark extraction.
- **NumPy**: Used for high-speed mathematical computations, coordinate geometry, matrix overlays, and boolean masks.

---

---

## 📈 Evaluation Metrics & Performance (How we measure success)

To prove the project is "Production-Ready", we use a dedicated evaluation script (`evaluate.py`) that calculates real-time performance data.

### 1. Frames Per Second (FPS)
*   **The Metric:** The number of full render-cycles completed per second.
*   **Why it matters:** Teaching requires fluid, non-laggy movement. A standard of **24-30 FPS** is expected for human-eye persistence of vision.

### 2. Inference Latency (ms)
*   **The Metric:** The time taken (in milliseconds) for the **MediaPipe model** to process a single frame.
*   **Why it matters:** Lower latency means the "ink" follows your finger instantly. Our system typically achieves **15-25ms**, which is faster than the human blink (~100ms).

### 3. Stability / Jitter Score
*   **The Metric:** The **Standard Deviation ($\sigma$)** of the index-finger coordinates $(x, y)$ when held theoretically still.
*   **Why it matters:** High jitter makes drawing impossible. Our stabilization logic (Euclidean Distance Filtering) keeps this score extremely low, ensuring smooth lines.

### 4. Gesture Accuracy (True Positive Rate)
*   **The Metric:** The percentage of times the system correctly identifies a "Draw" vs "Hover" vs "Erase" gesture.
*   **Target:** $>95\%$ accuracy across different lighting conditions and hand sizes.

---

## 🔬 How to Run the Evaluation
You can demonstrate the system's "Tech Stats" live by running:
```powershell
python evaluate.py
```
**Controls in Eval Mode:**
*   `f`: Starts a **10-second Benchmark** for FPS and Latency.
*   `j`: Starts a **5-second Stability Test** (Hold your finger still).
*   `g`: Starts a **Gesture Accuracy Check** (Verify each hand-sign).

---

## 💡 Quick QA Cheat Sheet
- **Q: Why MediaPipe over Haar Cascades?**
  *A: Haar cascades only provide a bounding box and are prone to lighting errors. MediaPipe provides 21 exact 3D spatial points (landmarks) on the hand, allowing for complex gesture math.*
- **Q: How does the Undo/Redo work efficiently without freezing the system?**
  *A: Instead of saving entire images to the hard drive, it saves lightweight NumPy arrays (matrices) to a bounded memory stack, making state transitions practically instantaneous.*
