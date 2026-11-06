# 🌌 Smart Air Writer & Spatial Whiteboard v6.0 PRO

An advanced, touchless **Spatial Air-Writing & Digital Whiteboard** system powered by Computer Vision, Deep Learning, and Geometric Splines. Write, sketch, drag, manipulate words, and export vector graphics in the air using only a standard webcam.

100% Free, Open-Source, and runs entirely locally without any paid APIs or cloud dependencies.

---

## ✨ Key Features

### 1. ✍️ Hybrid Dual-Mode Gesture Writing Engine
* **☝️ Index Pointing (0.8s Dwell)**: Point your index finger to begin drawing. A circular progress ring animates around your fingertip during the 0.8s dwell countdown before ink starts flowing.
* **✊ Fist / All Fingers Curled**: Instantly lifts the pen (`Pen Up`) and finishes the current stroke, committing it to the active word.
* **👌 Instant Pinch-to-Write**: Pinch your thumb and index together for immediate, zero-latency drawing.
* **✌️ Two-Finger Hover**: Raise index and middle fingers (peace sign) to move the cursor across the canvas freely without leaving any ink trail.
* **🖐️ Left-Hand Holographic Radial Palette**: Open your non-dominant (left) palm to trigger a floating 3D tool menu in orbit around your hand (color selection, brush switching, background toggle, sponge eraser, undo, and clear).
* **👌 Spatial Word Manipulation**: Pinch over any committed word to grab and drag it across the board (`offset_x`, `offset_y`), or fling it off-screen to delete.
* **👍 Thumbs-Up Commit (0.4s)**: Hold a thumbs-up gesture for 0.4s to commit the active word, append a space, and trigger auto-learning.

### 2. 🧠 Open-Source Cursive & Word-Level AI OCR
* **Local EasyOCR Integration (ResNet + LSTM + CTC)**: Unlike legacy character-by-character classifiers, EasyOCR recognizes **connected cursive words** and continuous handwriting in a single pass.
* **Asynchronous Multithreaded Processing**: OCR inference runs in a background daemon thread, guaranteeing that the webcam video stream remains buttery-smooth at 60 FPS.
* **Fast SymSpell $O(1)$ Correction & Bigram Context**: Symmetric delete lookup resolves handwriting ambiguities (`joue` $\rightarrow$ `JOUE`, `AZ1Z` $\rightarrow$ `AZIZ`, `2O25` $\rightarrow$ `2025`) using contextual vocabulary and bigram transition probabilities $P(w_i \mid w_{i-1})$.
* **Multi-Tier Fallback**: EasyOCR $\rightarrow$ EMNIST-CNN $\rightarrow$ Synthetic Net $\rightarrow$ Geometric Rules.

### 3. 🎨 Vector Spline Rendering & Smoothing
* **1€ Filter (One Euro Filter)**: Adaptive low-pass filtering on hand landmarks eliminates stationary jitter while preserving instant dynamic response during rapid handwriting strokes.
* **Catmull-Rom to Cubic Bézier Conversion**: Raw discrete points are analytically converted into continuous $C^1$ cubic Bézier curves for smooth anti-aliased in-app rendering.
* **Multi-Brush Aesthetics**: Neon Glow, Calligraphy (angle-sensitive stroke width), Highlighter (semi-transparent overlay), and Dotted.
* **Scalable Vector Export (`[P]`)**: Exports the entire board to a clean SVG vector file (`<path d="M... C...">`) with embedded glow filters, ready for Figma or Adobe Illustrator.
* **Full HD MP4 Recording (`[V]`)**: Record live sessions directly to MP4 video.

---

## 🚧 Active Development & Upcoming Roadmap

The following features are currently under active development:

- [ ] **Sequential Online Trajectory Recognition (1D-CNN / CTC)**: Training a dedicated temporal model directly on 6D motor stroke features (`[norm_x, norm_y, dx, dy, speed, curvature]`) collected via the interactive calibration tool (`[K]`). This will eliminate the rasterization step and recognize cursive words directly from gesture dynamics.
- [ ] **WebAssembly & ONNX Web Runtime**: Porting the online temporal tracker and gesture engine to the companion web application (`web-app/`) using `onnxruntime-web` for zero-install, in-browser spatial computing.
- [ ] **Virtual Z-Depth Plane Writing**: Integrating depth estimation so users can begin writing by pushing their finger forward through a virtual spatial glass plane, as an alternative to the 0.8s dwell timer.
- [ ] **Expanded Multilingual Corpus**: Expanding built-in dictionary indexes for Spanish, German, and Arabic handwriting with real-time vocabulary ingestion.

---

## 🖐️ Gesture Controls Quick Reference

| Gesture | Action | Description |
|---|---|---|
| ☝️ **Point Index (0.8s)** | **Air Writing** | Circular dwell ring fills for 0.8s, then writes continuous ink |
| ✊ **Closed Fist** | **Pen Up / Stop** | Stops writing immediately and finalizes stroke |
| ✌️ **Two Fingers (✌️)** | **Hover Cursor** | Move cursor across the board without drawing |
| 👌 **Pinch (Thumb+Index)** | **Pinch-to-Write / Grab** | Write instantly, or pinch an existing word to move / fling to delete |
| 👍 **Thumbs Up (0.4s)** | **Commit Word / Space** | Finalizes word recognition, inserts space, and triggers learning |
| 🖐️ **Left Open Palm** | **Radial Tool Menu** | Floating holographic palette (Colors, Brushes, Undo, Clear) |
| 🖐️ **Palm Held on Ink** | **Sponge Eraser** | Erases ink strokes directly beneath the palm area |

---

## ⌨️ Desktop Keyboard Shortcuts

| Key | Function |
|---|---|
| `[Space]` | Commit current word & add space |
| `[Tab]` | Cycle through alternate OCR word suggestions |
| `[Bksp]` | Delete last committed character |
| `[K]` | Toggle **Online Trajectory Calibration Session** (A-Z, 0-9) |
| `[N]` | Cycle Recognition Mode (**AUTO** / **DIGITS ONLY** / **LETTERS ONLY**) |
| `[B]` | Cycle Brushes (Neon, Calligraphy, Highlighter, Rainbow, Dotted) |
| `[W]` | Cycle Backgrounds (Webcam, Digital Whiteboard, Dark Studio) |
| `[P]` | Export clean vector **SVG (Cubic Bézier)** to `exports/` |
| `[V]` | Start / Stop HD MP4 Session Video Recording |
| `[Z]` | Undo last stroke or word |
| `[C]` | Clear canvas and reset text |
| `[1-5]` | Quick Color Switch (Cyan, Green, Yellow, Pink, White) |
| `[Q]` | Quit Application |

---

## 📁 Repository Structure

```
├── air_keyboard.py         # Main application loop, HUD, and video compositing
├── gesture_engine.py       # 1€ Filter, scale-adaptive pose estimator, FSM state machine
├── spline_renderer.py      # Catmull-Rom to Cubic Bézier math & SVG vector exporter
├── bimanual_ui.py          # Holographic radial menu & spatial WordManipulator
├── symspell_corrector.py   # SymSpell O(1) dictionary + Bigram language model
├── online_tracker.py       # 6D temporal trajectory processor & calibration engine
├── text_corrector.py       # Sequence-level probability corrector
├── train_model.py          # Offline EMNIST training pipeline (TensorFlow & scikit-learn)
├── requirements.txt        # Python dependency manifest
├── hand_landmarker.task    # MediaPipe hand tracking model
├── model/                  # Local neural network models
├── dataset/                # Calibration logs and trajectory datasets
├── exports/                # Saved vector SVGs and MP4 video recordings
└── web-app/                # Companion Web / React client implementation
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10 or 3.11
* A working webcam

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/aziz11414/air-keyboard.git
cd air-keyboard
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python air_keyboard.py
```

Show your hand to the camera to begin air-writing!

---

## 📜 License
This project is open-source and released under the [MIT License](LICENSE).
