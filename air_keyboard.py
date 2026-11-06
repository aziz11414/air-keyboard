import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math
import threading
from collections import deque
from datetime import datetime
from text_corrector import TextCorrector
from gesture_engine import OneEuroFilter, PointFilter2D, HandPoseEstimator, GestureStateMachine
from spline_renderer import render_stroke_smooth, export_svg_bezier
from bimanual_ui import RadialMenu, WordManipulator
from symspell_corrector import SymSpellCorrector
from online_tracker import OnlineTrajectory, CalibrationManager
from shape_engine import ShapeRecognizer
from session_replay import SessionReplayer
from smart_assistant import SessionAssistant
from virtual_camera import VirtualCameraManager
from multilingual_engine import MultilingualManager

# --- CONFIGURATION -----------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 576

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
EXPORTS_DIR = os.path.join(SCRIPT_DIR, "exports")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")        # created by train_model.py
os.makedirs(EXPORTS_DIR, exist_ok=True)

BRUSH_TYPES = ["NEON", "CALLIGRAPHY", "HIGHLIGHTER", "RAINBOW", "DOTTED"]
CURRENT_BRUSH_IDX = 0

BG_MODES = ["WEBCAM", "WHITEBOARD", "DARK"]
CURRENT_BG_IDX = 0

PEN_THICKNESS = 6
ACTIVATION_DELAY_SEC = 0.8   # délai d'activation index pour démarrer l'écriture
ERASER_RADIUS = 65
SMOOTH_FACTOR = 0.55

PALM_HOLD_SEC = 0.5          # palm sponge eraser engages after 0.5s hold

PALETTE = {
    "1": ("Neon Cyan", (255, 235, 0)),
    "2": ("Electric Green", (0, 255, 128)),
    "3": ("Vibrant Yellow", (0, 230, 255)),
    "4": ("Hot Pink", (200, 50, 255)),
    "5": ("Pure White", (255, 255, 255)),
}
CURRENT_COLOR_KEY = "1"
HAND_2_COLOR = (200, 50, 255)


# --- STROKE & WORD DATA STRUCTURE --------------------------------------
class StrokeItem:
    def __init__(self, points, color, thickness, brush="NEON"):
        self.points = list(points)
        self.color = color
        self.thickness = thickness
        self.brush = brush


class WordItem:
    def __init__(self, word_id):
        self.id = word_id
        self.strokes = []
        self.text = ""
        self.offset_x = 0
        self.offset_y = 0
        self.timestamp = time.time()

    def add_stroke(self, stroke):
        self.strokes.append(stroke)

    def get_bbox(self):
        all_pts = []
        for s in self.strokes:
            for p in s.points:
                all_pts.append((p[0] + self.offset_x, p[1] + self.offset_y))
        if not all_pts:
            return None
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return min_x, min_y, max(1, max_x - min_x), max(1, max_y - min_y)

    def is_empty(self):
        return len(self.strokes) == 0


class DocumentManager:
    """Hierarchical list of words/strokes with undo/redo and stroke eraser."""

    def __init__(self):
        self.committed_words = []
        self.current_word = WordItem(1)
        self.redo_stack = []
        self.next_word_id = 2

    def add_stroke_to_current(self, stroke):
        self.current_word.add_stroke(stroke)
        self.redo_stack.clear()

    def commit_current_word(self, recognized_text=""):
        if not self.current_word.is_empty():
            self.current_word.text = recognized_text
            self.committed_words.append(self.current_word)
            self.current_word = WordItem(self.next_word_id)
            self.next_word_id += 1
            return True
        return False

    def erase_at(self, center, radius):
        """Remove every stroke that has a point inside the circle. Returns count removed."""
        cx, cy = center
        removed = 0
        for w in self.committed_words + [self.current_word]:
            ox, oy = w.offset_x, w.offset_y
            keep = []
            for s in w.strokes:
                hit = any(math.hypot(p[0] + ox - cx, p[1] + oy - cy) <= radius for p in s.points)
                if hit:
                    removed += 1
                else:
                    keep.append(s)
            w.strokes = keep
        self.committed_words = [w for w in self.committed_words if not w.is_empty()]
        return removed

    def undo(self):
        if not self.current_word.is_empty():
            popped = self.current_word.strokes.pop()
            self.redo_stack.append(('stroke', popped))
            return "STROKE"
        elif self.committed_words:
            popped_word = self.committed_words.pop()
            self.redo_stack.append(('word', popped_word))
            return "WORD"
        return None

    def redo(self):
        if self.redo_stack:
            item_type, item = self.redo_stack.pop()
            if item_type == 'stroke':
                self.current_word.add_stroke(item)
            elif item_type == 'word':
                self.committed_words.append(item)
            return True
        return False

    def clear_all(self):
        self.committed_words.clear()
        self.current_word = WordItem(self.next_word_id)
        self.redo_stack.clear()

    def render(self, canvas):
        canvas.fill(0)
        all_words = self.committed_words + ([self.current_word] if not self.current_word.is_empty() else [])

        for w in all_words:
            ox, oy = w.offset_x, w.offset_y
            for s in w.strokes:
                pts = [(int(p[0] + ox), int(p[1] + oy)) for p in s.points]
                render_stroke_smooth(canvas, pts, s.color, s.thickness, s.brush)

    def export_svg(self, arg1, arg2, arg3=None):
        if isinstance(arg1, str):
            filepath, width, height = arg1, arg2, arg3
        else:
            width, height, filepath = arg1, arg2, arg3
        all_strokes = []
        all_words = self.committed_words + ([self.current_word] if not self.current_word.is_empty() else [])
        for w in all_words:
            ox, oy = w.offset_x, w.offset_y
            for s in w.strokes:
                pts = [(p[0] + ox, p[1] + oy) for p in s.points]
                shifted_stroke = StrokeItem(pts, s.color, s.thickness, s.brush)
                all_strokes.append(shifted_stroke)
        export_svg_bezier(all_strokes, width, height, filepath)


# --- WORD OCR ENGINE (EMNIST-CNN / EMNIST-MLP / SYNTH-NET / RULES) -----
class WordOCREngine:
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # label order 0..35

    def __init__(self, mode="auto"):
        self.mode = mode   # "auto" (default: words & digits), "digits", "letters"
        self.backend_name = "RULES"
        self.easyocr_reader = None
        self.tf_model = None
        self.sk_model = None
        self.sk_scaler = None
        self.net = None
        self.char_templates = self._build_templates()
        self._load_best_model()

    @property
    def allow_digits(self):
        return self.mode != "letters"

    @allow_digits.setter
    def allow_digits(self, val):
        self.mode = "digits" if val else "auto"

    # ------------------------------------------------------------------
    # Model loading: EMNIST-CNN > EMNIST-MLP > SYNTH-NET (auto) > RULES
    # ------------------------------------------------------------------
    def _model_dirs(self):
        dirs = [MODEL_DIR, os.path.join(os.getcwd(), "model")]
        out = []
        for d in dirs:
            if d not in out:
                out.append(d)
        return out

    def _load_best_model(self):
        dirs = self._model_dirs()

        # 0. Open-Source EasyOCR (Cursive, full words, 100% free & offline)
        try:
            import easyocr
            self.easyocr_reader = easyocr.Reader(['fr', 'en'], gpu=False, verbose=False)
            self.backend_name = "EasyOCR"
            print("[OCR] Active engine: EasyOCR (Open-Source, Local Cursive & Words)")
        except Exception as e:
            print(f"[OCR] EasyOCR unavailable: {e}")
            self.easyocr_reader = None

        # 1. TensorFlow CNN (fallback for single isolated characters)
        for d in dirs:
            for name in ("emnist_model.keras", "emnist_model.h5"):
                m_path = os.path.join(d, name)
                if os.path.exists(m_path):
                    try:
                        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
                        import tensorflow as tf
                        self.tf_model = tf.keras.models.load_model(m_path)
                        if self.backend_name != "EasyOCR":
                            self.backend_name = "EMNIST-CNN"
                        print(f"[OCR] Secondary/Fallback engine: EMNIST-CNN ({m_path})")
                        break
                    except Exception as e:
                        pass
            if self.tf_model is not None:
                break

        if self.backend_name == "EasyOCR" or self.backend_name == "EMNIST-CNN":
            return

        # 2. scikit-learn MLP trained by train_model.py --sklearn
        for d in dirs:
            path = os.path.join(d, "emnist_sklearn.pkl")
            if os.path.exists(path):
                try:
                    import pickle
                    with open(path, "rb") as f:
                        bundle = pickle.load(f)
                    self.sk_model = bundle["model"]
                    self.sk_scaler = bundle.get("scaler")
                    self.backend_name = "EMNIST-MLP"
                    print(f"[OCR] Active engine: EMNIST-MLP ({path})")
                    return
                except Exception as e:
                    print(f"[OCR] Could not load {path}: {e}")

        print("[OCR] No EMNIST model found. Looked in:")
        for d in dirs:
            print(f"        {d}")
        print("[OCR] For the best accuracy run:  python train_model.py")

        # 3. Built-in model trained once on synthetic block letters (numpy only)
        if self._load_or_train_synth():
            return

        # 4. Last resort: geometric rules
        print("[OCR] Active engine: RULES (basic, often wrong)")

    # ------------------------------------------------------------------
    # Built-in synthetic model (needs scikit-learn only)
    # ------------------------------------------------------------------
    def _load_or_train_synth(self):
        path = os.path.join(MODEL_DIR, "synthetic_net.npz")
        if os.path.exists(path):
            try:
                d = np.load(path)
                self.net = {k: d[k] for k in ("W1", "b1", "W2", "b2", "mu", "sd")}
                self.backend_name = "SYNTH-NET"
                print(f"[OCR] Active engine: SYNTH-NET ({path})")
                return True
            except Exception as e:
                print(f"[OCR] Could not read {path}: {e} (retraining)")

        print("[OCR] Training the built-in letter model (one time only, ~30 s)...")
        try:
            X, y = self._make_synthetic_dataset()
            self.net = self._train_numpy_net(X, y)
            os.makedirs(MODEL_DIR, exist_ok=True)
            np.savez(path, **self.net)
            self.backend_name = "SYNTH-NET"
            print(f"[OCR] Active engine: SYNTH-NET (saved to {path})")
            return True
        except Exception as e:
            print(f"[OCR] Built-in training failed: {e}")
            return False

    def _train_numpy_net(self, X, y, hidden=384, epochs=50, batch=128, lr=2e-3, seed=1):
        """Tiny 784-384-36 network trained with Adam, numpy only."""
        rng = np.random.RandomState(seed)
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 0.1
        Xn = ((X - mu) / sd).astype(np.float32)
        n = len(Xn)
        idx = rng.permutation(n)
        n_val = n // 10
        val_i, tr_i = idx[:n_val], idx[n_val:]

        W1 = (rng.randn(784, hidden) * np.sqrt(2.0 / 784)).astype(np.float32)
        b1 = np.zeros(hidden, np.float32)
        W2 = (rng.randn(hidden, 36) * np.sqrt(2.0 / hidden)).astype(np.float32)
        b2 = np.zeros(36, np.float32)
        params = [W1, b1, W2, b2]
        m = [np.zeros_like(q) for q in params]
        v = [np.zeros_like(q) for q in params]
        t = 0

        for ep in range(epochs):
            order = rng.permutation(tr_i)
            for i in range(0, len(order), batch):
                bi = order[i:i + batch]
                xb, yb = Xn[bi], y[bi]
                xb = xb + rng.randn(*xb.shape).astype(np.float32) * 0.05   # light noise
                z1 = xb @ W1 + b1
                a1 = np.maximum(z1, 0)
                z2 = a1 @ W2 + b2
                z2 -= z2.max(axis=1, keepdims=True)
                p = np.exp(z2)
                p /= p.sum(axis=1, keepdims=True)
                g2 = p
                g2[np.arange(len(bi)), yb] -= 1.0
                g2 /= len(bi)
                gW2 = a1.T @ g2 + 1e-4 * W2
                gb2 = g2.sum(axis=0)
                g1 = (g2 @ W2.T) * (z1 > 0)
                gW1 = xb.T @ g1 + 1e-4 * W1
                gb1 = g1.sum(axis=0)
                t += 1
                for k, (q, g) in enumerate(zip(params, [gW1, gb1, gW2, gb2])):
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = 0.999 * v[k] + 0.001 * g * g
                    mh = m[k] / (1 - 0.9 ** t)
                    vh = v[k] / (1 - 0.999 ** t)
                    q -= lr * mh / (np.sqrt(vh) + 1e-8)

        net = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
        pv = self._net_forward(net, X[val_i])
        print(f"[OCR] Built-in model validation accuracy: {np.mean(pv.argmax(axis=1) == y[val_i]):.1%}")
        return net

    @staticmethod
    def _net_forward(net, X):
        xn = (X - net["mu"]) / net["sd"]
        a1 = np.maximum(xn @ net["W1"] + net["b1"], 0)
        z2 = a1 @ net["W2"] + net["b2"]
        z2 -= z2.max(axis=1, keepdims=True)
        p = np.exp(z2)
        return p / p.sum(axis=1, keepdims=True)

    def _make_synthetic_dataset(self, per_font=90, seed=0):
        rng = np.random.RandomState(seed)
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
                 cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL]
        X, y = [], []
        for label, ch in enumerate(self.CHARS):
            for font in fonts:
                for _ in range(per_font):
                    img = self._render_synthetic_char(ch, font, rng)
                    X.append(self.preprocess_to_emnist(img).reshape(-1))
                    y.append(label)
        return np.array(X, dtype=np.float32), np.array(y)

    def _render_synthetic_char(self, ch, font, rng):
        size = 160
        img = np.zeros((size, size), dtype=np.uint8)
        scale = 3.0 + rng.rand() * 1.5
        thick = int(rng.randint(2, 6))
        (tw, th), base = cv2.getTextSize(ch, font, scale, thick)
        cv2.putText(img, ch, ((size - tw) // 2, (size + th) // 2), font, scale, 255, thick)

        # Hand-like wobble: smooth random displacement field
        mag = 1.5 + rng.rand() * 4.0
        dx = cv2.GaussianBlur(rng.randn(size, size).astype(np.float32), (0, 0), 9)
        dy = cv2.GaussianBlur(rng.randn(size, size).astype(np.float32), (0, 0), 9)
        dx = dx / (dx.std() + 1e-6) * mag
        dy = dy / (dy.std() + 1e-6) * mag
        gx, gy = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
        img = cv2.remap(img, gx + dx, gy + dy, cv2.INTER_LINEAR)

        # Random rotation / shear / stretch
        ang = np.deg2rad(rng.uniform(-12, 12))
        shear = rng.uniform(-0.25, 0.25)
        sx, sy = rng.uniform(0.75, 1.25), rng.uniform(0.85, 1.15)
        c, s = np.cos(ang), np.sin(ang)
        A = np.array([[c * sx, -s + shear, 0], [s, c * sy, 0]], dtype=np.float32)
        A[:, 2] = np.array([size / 2, size / 2]) - A[:, :2] @ np.array([size / 2, size / 2])
        img = cv2.warpAffine(img, A, (size, size))
        _, img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
        return img

    # ------------------------------------------------------------------
    # Preprocessing: character patch -> EMNIST-like 28x28
    # ------------------------------------------------------------------
    def preprocess_to_emnist(self, patch):
        ys, xs = np.nonzero(patch)
        if len(ys) == 0:
            return np.zeros((28, 28), dtype=np.float32)

        # Tight crop around ink
        crop = patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        ch_h, ch_w = crop.shape
        if ch_h == 0 or ch_w == 0:
            return np.zeros((28, 28), dtype=np.float32)

        # Fit into 20x20 box preserving aspect ratio
        s = 20.0 / max(ch_h, ch_w)
        nw, nh = max(1, int(round(ch_w * s))), max(1, int(round(ch_h * s)))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

        # Place in 28x28 canvas
        out = np.zeros((28, 28), dtype=np.uint8)
        oy, ox = (28 - nh) // 2, (28 - nw) // 2
        out[oy:oy + nh, ox:ox + nw] = resized

        # Center-of-mass alignment
        m = cv2.moments(out)
        if m["m00"] > 0:
            shift_x = float(np.clip(14.0 - m["m10"] / m["m00"], -4, 4))
            shift_y = float(np.clip(14.0 - m["m01"] / m["m00"], -4, 4))
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            out = cv2.warpAffine(out, M, (28, 28), borderValue=0)

        # Normalize to [0.0, 1.0]
        return (out.astype(np.float32) / 255.0)

    # ------------------------------------------------------------------
    # Prediction (all characters of a word in one batch)
    # ------------------------------------------------------------------
    def _scores_from_probs(self, probs, n_classes):
        """Return one score per CHARS entry (0-9, A-Z).
        36 classes: 0-9 and A-Z.
        26 classes: A-Z only.
        62 classes: EMNIST byclass."""
        scores = np.zeros(36, dtype=np.float32)
        if n_classes == 36:
            scores[:36] = probs[:36]
        elif n_classes == 26:
            scores[10:36] = probs[:26]
        elif n_classes >= 62:
            scores[:10] = probs[:10]
            scores[10:36] = probs[10:36] + probs[36:62]
        else:
            scores[:min(36, n_classes)] = probs[:36]
        return scores

    def _predict_chars(self, char_patches):
        norm_patches = [self.preprocess_to_emnist(cp) for cp in char_patches]
        X = np.stack(norm_patches)
        if self.tf_model is not None:
            try:
                probs = self.tf_model(X.reshape(-1, 28, 28, 1), training=False).numpy()
            except Exception:
                probs = self.tf_model.predict(X.reshape(-1, 28, 28, 1), verbose=0)
            n_classes = probs.shape[1]
        elif self.net is not None:
            probs = self._net_forward(self.net, X.reshape(len(X), 784))
            n_classes = 36
        else:
            flat = X.reshape(len(X), 784)
            if self.sk_scaler is not None:
                flat = self.sk_scaler.transform(flat)
            part = self.sk_model.predict_proba(flat)
            n_classes = 62 if self.backend_name == "EMNIST-MLP" else 36
            probs = np.zeros((len(X), n_classes), dtype=np.float32)
            probs[:, self.sk_model.classes_] = part

        text = ""
        char_candidates = []
        for idx, p in enumerate(probs):
            sc = self._scores_from_probs(p, n_classes)
            if self.mode == "letters":
                sc[:10] = -1.0
            elif self.mode == "digits":
                sc[10:] = -1.0

            best_idx = int(np.argmax(sc))
            best_c = self.CHARS[best_idx]
            top_indices = np.argsort(sc)[::-1][:4]

            # --- Disambiguation heuristics for handwriting ---
            patch28 = norm_patches[idx]

            # 1. H vs G: G has rounded top roof and bottom floor
            if best_c == 'H':
                top_center = np.sum(patch28[2:7, 10:18] > 0.1)
                bot_center = np.sum(patch28[21:26, 10:18] > 0.1)
                if top_center >= 14 and bot_center >= 16:
                    best_c = 'G'

            # 2. B vs E: In B, right loops are closed (holes exist). In E, right side is open.
            if best_c == 'B':
                u8 = (patch28 * 255).astype(np.uint8)
                _, hierarchy = cv2.findContours(u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                holes = 0
                if hierarchy is not None:
                    for h in hierarchy[0]:
                        if h[3] != -1:
                            holes += 1
                if holes == 0:
                    mid_right = np.sum(patch28[8:12, 18:24] > 0.1)
                    if mid_right == 0 and 'E' in [self.CHARS[k] for k in top_indices[:3]]:
                        best_c = 'E'

            # 3. I / 1 / 7 vs Y: In Y, top has two distinct branches with hollow center
            if best_c in ('I', '1', '7', 'T'):
                top_ink = patch28[2:8, :] > 0.1
                top_cols = np.where(top_ink)[1]
                if len(top_cols) > 0 and (top_cols.max() - top_cols.min()) >= 8:
                    top_center = np.sum(patch28[2:6, 12:16] > 0.1)
                    if top_center == 0:
                        best_c = 'Y'

            # 4. Y vs N disambiguation:
            # In 'Y', bottom ink is centered and narrow (single stem).
            # In 'N', bottom ink has two legs on the left and right (span >= 9).
            if best_c == 'Y':
                bot_ink = patch28[18:24, :] > 0.1
                bot_cols = np.where(bot_ink)[1]
                span = (bot_cols.max() - bot_cols.min()) if len(bot_cols) else 0
                if span >= 9:
                    best_c = 'N'

            # 5. W / X vs H: Arch-style 'h' has tall left stem and zero top-right ink
            if best_c in ('W', 'X'):
                top_right_ink = np.sum(patch28[2:8, 16:26] > 0.1)
                top_left_ink = np.sum(patch28[2:8, 2:12] > 0.1)
                if top_right_ink < 4 and top_left_ink > 12:
                    best_c = 'H'

            # 6. Q vs A/O disambiguation:
            # 'Q' must have a distinct bottom-right diagonal tail.
            # If it has a pointy top apex and no tail, it is an 'A'.
            if best_c == 'Q' and self.mode != "digits":
                br_tail = np.sum(patch28[16:24, 16:24] > 0.1)
                top_apex = np.sum(patch28[2:8, 10:18] > 0.1)
                if top_apex > 5 and br_tail < 8:
                    best_c = 'A'

            # 7. O vs Q disambiguation:
            if best_c == 'O':
                tail_ink = np.sum(patch28[18:26, 18:26] > 0.1)
                if tail_ink >= 12 and 'Q' in [self.CHARS[k] for k in top_indices[:3]]:
                    best_c = 'Q'

            text += best_c

            # Assemble top candidates for this character position
            cands = [(best_c, float(max(0.60, sc[best_idx])))]
            for k in top_indices:
                ch = self.CHARS[k]
                if ch != best_c and ch not in [c[0] for c in cands]:
                    prob_val = float(max(0.01, sc[k]))
                    cands.append((ch, prob_val))
            char_candidates.append(cands[:4])

        return text, char_candidates

    # ------------------------------------------------------------------
    # Word -> characters
    # ------------------------------------------------------------------
    def recognize_word_item(self, word_item, corrector=None, mode="auto"):
        bbox = word_item.get_bbox()
        if not bbox:
            return "", []

        x, y, w, h = bbox
        pad = 25
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(FRAME_WIDTH, x + w + pad), min(FRAME_HEIGHT, y + h + pad)

        patch_w = x1 - x0
        patch_h = y1 - y0
        if patch_w <= 0 or patch_h <= 0:
            return "", []

        # --- Render strokes with clean lines ---
        patch = np.zeros((patch_h, patch_w), dtype=np.uint8)
        ox, oy = word_item.offset_x, word_item.offset_y
        for s in word_item.strokes:
            pts = [(p[0] + ox - x0, p[1] + oy - y0) for p in s.points]
            th = max(s.thickness, 4)
            for i in range(1, len(pts)):
                cv2.line(patch, pts[i - 1], pts[i], 255, th)
                cv2.circle(patch, pts[i], th // 2, 255, -1)

        # 1. EasyOCR full-word recognition (handles cursive, connected writing & whole words)
        if getattr(self, "easyocr_reader", None) is not None:
            try:
                pad_val = 20
                patch_padded = cv2.copyMakeBorder(patch, pad_val, pad_val, pad_val, pad_val, cv2.BORDER_CONSTANT, value=0)
                ph, pw = patch_padded.shape[:2]
                if ph < 80:
                    scale_f = 80.0 / ph
                    patch_padded = cv2.resize(patch_padded, (int(pw * scale_f), 80), interpolation=cv2.INTER_LINEAR)
                results = self.easyocr_reader.readtext(patch_padded, detail=0)
                if results:
                    raw_text = " ".join(results).strip().upper()
                    raw_text = "".join(c for c in raw_text if c.isalnum() or c == " ")
                    if raw_text:
                        if corrector is not None:
                            best_w, alts = corrector.correct(raw_text, mode=mode)
                            return best_w, alts
                        return raw_text, []
            except Exception as e:
                print(f"[OCR] EasyOCR error: {e}")

        # --- Fallback: Find contours and their bounding boxes for character-level OCR ---
        contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]
        if not boxes:
            return "", []

        boxes.sort(key=lambda b: b[0])

        # --- SMART MERGE: Only merge inner fragments (crossbars, dots) ---
        # Distinct letters (height > 35% of max height) are NEVER merged horizontally,
        # even if their x-bounds slightly overlap due to italic slant!
        max_h = max(b[3] for b in boxes)
        merged = []
        for bx, by, bw, bh in boxes:
            if not merged:
                merged.append([bx, by, bw, bh])
                continue

            px, py, pw, ph = merged[-1]

            # A sub-stroke is small in height and horizontally within the main stroke
            is_curr_sub = (bh < max_h * 0.35) and (bx >= px - 12) and (bx + bw <= px + pw + 12)
            is_prev_sub = (ph < max_h * 0.35) and (px >= bx - 12) and (px + pw <= bx + bw + 12)

            if is_curr_sub or is_prev_sub:
                nx = min(px, bx)
                ny = min(py, by)
                nw = max(px + pw, bx + bw) - nx
                nh = max(py + ph, by + bh) - ny
                merged[-1] = [nx, ny, nw, nh]
            else:
                merged.append([bx, by, bw, bh])

        # --- Extract character patches with a small padding ---
        char_patches = []
        for (bx, by, bw, bh) in merged:
            cp = max(3, int(min(bw, bh) * 0.08))
            py0 = max(0, by - cp)
            py1 = min(patch_h, by + bh + cp)
            px0 = max(0, bx - cp)
            px1 = min(patch_w, bx + bw + cp)
            char_patches.append(patch[py0:py1, px0:px1])

        if not char_patches:
            return "", []

        if self.backend_name != "RULES":
            try:
                raw_text, char_candidates = self._predict_chars(char_patches)
                if corrector is not None:
                    best_w, alts = corrector.correct(char_candidates, mode=mode)
                    return best_w, alts
                return raw_text, []
            except Exception as e:
                print(f"[OCR] Prediction error, using RULES: {e}")

        rule_text = "".join(self._classify_rules(cp) for cp in char_patches)
        return rule_text, []

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------
    def _classify_rules(self, patch):
        h, w = patch.shape
        if h == 0 or w == 0:
            return ""
        aspect = w / float(h)
        padded = cv2.copyMakeBorder(patch, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0)
        contours, hierarchy = cv2.findContours(padded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num_holes = 0
        if hierarchy is not None:
            for info in hierarchy[0]:
                if info[3] != -1:
                    num_holes += 1

        mid_y, mid_x = h // 2, w // 2
        horiz_crossings = self._crossings(patch[mid_y, :])
        top_density = np.sum(patch[:mid_y, :] > 0) / max(1, np.sum(patch > 0))

        char = ""
        if num_holes >= 2:
            char = "B" if aspect < 0.8 else ("8" if self.allow_digits else "B")
        elif num_holes == 1:
            if top_density > 0.60:
                char = "P" if horiz_crossings <= 2 else "A"
            elif top_density < 0.40:
                char = "Q"
            else:
                char = "O" if abs(aspect - 1.0) < 0.3 else "D"
        elif aspect < 0.28:
            char = "I"
        else:
            left_strip = patch[:, :max(1, w // 4)]
            right_strip = patch[:, -max(1, w // 4):]
            if np.sum(left_strip > 0) > (0.35 * h) and np.sum(right_strip > 0) > (0.35 * h):
                if horiz_crossings >= 2:
                    if np.sum(patch[mid_y - 2:mid_y + 3, w // 4:3 * w // 4] > 0) > (0.28 * w):
                        char = "H"
                    else:
                        char = "U" if np.sum(patch[mid_y:, :] > 0) > np.sum(patch[:mid_y, :] > 0) else "N"

            if not char:
                if np.sum(patch[:max(1, h // 4), :] > 0) > (0.55 * w) and np.sum(patch[:, mid_x] > 0) > (0.5 * h):
                    char = "T"
                elif np.sum(left_strip > 0) > (0.5 * h) and np.sum(patch[-max(1, h // 4):, :] > 0) > (0.5 * w):
                    char = "L"
                elif np.sum(left_strip > 0) > (0.5 * h) and np.sum(right_strip > 0) < (0.2 * h):
                    char = "C"
                elif horiz_crossings == 3:
                    char = "M"
                elif horiz_crossings == 2:
                    char = "N" if aspect < 0.9 else "W"

        if not char:
            resized = cv2.resize(patch, (24, 24)).astype(np.float32) / 255.0
            best_char, best_sc = "", -1.0
            for c, tmpl in self.char_templates.items():
                if not self.allow_digits and c.isdigit():
                    continue
                sc = np.sum(resized * tmpl) / (np.linalg.norm(resized) * np.linalg.norm(tmpl) + 1e-5)
                if sc > best_sc:
                    best_sc, best_char = sc, c
            char = best_char if best_sc > 0.42 else ""

        if not self.allow_digits and char.isdigit():
            char = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}.get(char, "")

        return char.upper()

    def _crossings(self, line_px):
        c, in_s = 0, False
        for p in line_px:
            if p > 0 and not in_s:
                c += 1
                in_s = True
            elif p == 0:
                in_s = False
        return c

    def _build_templates(self):
        tmpls = {}
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            img = np.zeros((36, 36), dtype=np.uint8)
            ts = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.putText(img, char, ((36 - ts[0]) // 2, (36 + ts[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
            tmpls[char] = cv2.resize(img, (24, 24)).astype(np.float32) / 255.0
        return tmpls


# --- MULTI-HAND TRACKER ------------------------------------------------
class MultiHandTracker:
    def __init__(self):
        if not os.path.exists(HAND_MODEL_PATH):
            raise FileNotFoundError(f"Model missing: {HAND_MODEL_PATH}")

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        self.latest_result = None

        def _cb(result, output_image, timestamp_ms):
            self.latest_result = result

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            num_hands=2,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.60,
            running_mode=RunningMode.LIVE_STREAM,
            result_callback=_cb,
        )

        self.landmarker = HandLandmarker.create_from_options(options)
        self.ts = 0

    def process(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.ts += 33
        self.landmarker.detect_async(mp_img, self.ts)

        if not self.latest_result or not self.latest_result.hand_landmarks:
            return []

        hands_pts = []
        for hand_lm in self.latest_result.hand_landmarks:
            pts = [(int(p.x * w), int(p.y * h)) for p in hand_lm]
            hands_pts.append(pts)
        return hands_pts

    def get_handedness(self):
        if not self.latest_result or not self.latest_result.handedness:
            return []
        return [h[0].category_name for h in self.latest_result.handedness]

    def classify_hand(self, pts):
        # Rotation-invariant classification
        gesture, conf, metrics = HandPoseEstimator.classify_hand_pro(pts)
        if gesture == "OPEN_PALM":
            return "ERASE"
        elif gesture == "TWO_FINGERS_HOVER":
            return "TWO_FINGERS"
        elif gesture == "INDEX_POINT":
            return "INDEX_UP"
        return gesture

    def get_palm_center(self, pts):
        px = int((pts[0][0] + pts[5][0] + pts[9][0] + pts[17][0]) / 4)
        py = int((pts[0][1] + pts[5][1] + pts[9][1] + pts[17][1]) / 4)
        return (px, py)

    def draw_skeleton(self, frame, pts, color=(100, 100, 100)):
        conns = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
        ]
        for s, e in conns:
            cv2.line(frame, pts[s], pts[e], color, 1)
        for p in pts:
            cv2.circle(frame, p, 3, (0, 255, 120), -1)

    def close(self):
        self.landmarker.close()


# --- MAIN APPLICATION --------------------------------------------------
def main():
    global CURRENT_COLOR_KEY, CURRENT_BRUSH_IDX, CURRENT_BG_IDX

    print("=" * 72)
    print("   SMART AIR WRITER & SPATIAL WHITEBOARD v6.0 PRO")
    print("=" * 72)

    # Load OCR & SymSpell Engine
    ocr = WordOCREngine()
    corrector = SymSpellCorrector(SCRIPT_DIR)
    radial_menu = RadialMenu()
    word_manipulator = WordManipulator()
    calibration_mgr = CalibrationManager(os.path.join(SCRIPT_DIR, "dataset"))
    one_euro_hands = [PointFilter2D(mincutoff=1.2, beta=0.008), PointFilter2D(mincutoff=1.2, beta=0.008)]
    gesture_fsms = [GestureStateMachine(), GestureStateMachine()]

    print("")
    print("  GESTURES PRO:")
    print("  Index Point         : INSTANT AIR WRITING (0.0s delay - Main Drawing Mode)")
    print("  Pinch (Thumb+Index) : Pinch-to-Write OR Grab & Move Word (or fling to delete)")
    print("  Two Fingers (Peace) : Hover Cursor (Pen Up / Move without drawing)")
    print("  Left Open Palm      : FLOATING RADIAL TOOL PALETTE (Colors/Brush/Undo)")
    print("  Thumbs Up (0.4s)    : Commit Word / Space [OK]")
    print("  Open Palm on ink    : Sponge Eraser under palm")
    print("")
    print("  KEYS:")
    print("  [K] Calibration Mode      | [Space] Commit Word       | [C] Clear All")
    print("  [R] Re-scan OCR           | [N] Mode (Auto/0-9/A-Z)   | [Z] Undo")
    print("  [Tab] Cycle Alternatives  | [Bksp] Delete Character   | [B] Brush")
    print("  [W] Board                 | [P] Export SVG (Bezier)   | [V] Record MP4")
    print("  [F] Auto-Shape Snap       | [Y] Session Replay        | [M] Export Notes")
    print("  [U] Virtual Cam (OBS)     | [L] Lang (FR/EN/AR)       | [1-5] Color | [Q] Quit")
    print("=" * 72)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    tracker = MultiHandTracker()
    doc = DocumentManager()

    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    hand_states = [
        {"stroke": [], "smooth_x": None, "smooth_y": None, "start_time": None, "active": False, "prev_pt": None, "thick": PEN_THICKNESS, "index_hold_start": None},
        {"stroke": [], "smooth_x": None, "smooth_y": None, "start_time": None, "active": False, "prev_pt": None, "thick": PEN_THICKNESS, "index_hold_start": None},
    ]

    gesture_history = deque(maxlen=6)
    palm_hold_start = None
    eraser_mode = False

    commit_hold_start = None
    commit_cooldown_until = 0.0
    COMMIT_HOLD_SEC = 0.40

    committed_text = ""
    live_word_text = ""
    live_word_alts = []
    current_best_raw = ""
    alt_idx = 0

    # Modern v6.0 PRO Extensions
    replayer = SessionReplayer()
    assistant = SessionAssistant(EXPORTS_DIR)
    vcam_mgr = VirtualCameraManager(FRAME_WIDTH, FRAME_HEIGHT, fps=30)
    lang_mgr = MultilingualManager("FR")
    auto_shapes_enabled = True
    last_frame_time = time.time()

    def finalize_hand_stroke(h_state, draw_color, curr_brush):
        nonlocal canvas
        if not h_state["stroke"]:
            return
        if eraser_mode:
            h_state["stroke"] = []
            return

        if calibration_mgr.is_active:
            calibration_mgr.record_stroke(h_state["stroke"])
            char_done = calibration_mgr.CALIBRATION_CHARS[max(0, calibration_mgr.char_index - 1)]
            show_toast(f"CALIBRATED: '{char_done}'", (0, 255, 180), 1.5)
        else:
            final_pts = h_state["stroke"]
            # Auto-Shape Detection & Snapping
            if auto_shapes_enabled and len(final_pts) >= 8:
                shape_type, snapped_pts, shape_conf = ShapeRecognizer.recognize_and_snap(final_pts)
                if shape_type != "NONE" and shape_conf >= 0.85:
                    final_pts = [(pt[0], pt[1], time.time()) for pt in snapped_pts]
                    assistant.add_shape(shape_type)
                    show_toast(f"{lang_mgr.get('SNAPPED')}: {shape_type}", (0, 255, 180), 2.0)

            stroke_item = StrokeItem(
                points=final_pts,
                color=draw_color,
                thickness=PEN_THICKNESS,
                brush=curr_brush
            )
            doc.add_stroke_to_current(stroke_item)
            replayer.record_stroke(stroke_item)
            doc.render(canvas)
            update_live_ocr()

        h_state["stroke"] = []

    ocr_busy = False

    def update_live_ocr():
        nonlocal live_word_text, live_word_alts, current_best_raw, alt_idx, ocr_busy
        if doc.current_word.is_empty():
            live_word_text = ""
            live_word_alts = []
            current_best_raw = ""
            alt_idx = 0
            return

        # Snapshot current word for thread safety
        word_copy = WordItem(doc.current_word.id)
        word_copy.strokes = [StrokeItem(s.points, s.color, s.thickness, s.brush) for s in doc.current_word.strokes]
        word_copy.offset_x = doc.current_word.offset_x
        word_copy.offset_y = doc.current_word.offset_y
        m_mode = ocr.mode

        def _worker():
            nonlocal live_word_text, live_word_alts, current_best_raw, alt_idx, ocr_busy
            ocr_busy = True
            try:
                res_word, res_alts = ocr.recognize_word_item(word_copy, corrector=corrector, mode=m_mode)
                if res_word:
                    live_word_text = res_word
                    current_best_raw = res_word
                    live_word_alts = res_alts
                    alt_idx = 0
            finally:
                ocr_busy = False

        if not ocr_busy:
            t = threading.Thread(target=_worker, daemon=True)
            t.start()

    is_recording = False
    video_writer = None
    record_start_time = 0.0
    rainbow_hue = 0

    toast_msg = ""
    toast_col = (0, 255, 128)
    toast_exp = 0.0

    def show_toast(msg, col=(0, 255, 128), dur=1.8):
        nonlocal toast_msg, toast_col, toast_exp
        toast_msg, toast_col, toast_exp = msg, col, time.time() + dur

    def get_background(frame, mode):
        if mode == "WEBCAM":
            return frame.copy()
        elif mode == "WHITEBOARD":
            return np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 250, dtype=np.uint8)
        elif mode == "DARK":
            # Plain dark background, no grid squares
            return np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (22, 22, 28), dtype=np.uint8)
        return frame.copy()

    print("Camera active! Show your index finger to begin.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            now = time.time()
            dt = max(0.001, min(0.1, now - last_frame_time))
            last_frame_time = now
            curr_brush = BRUSH_TYPES[CURRENT_BRUSH_IDX]
            curr_bg = BG_MODES[CURRENT_BG_IDX]

            # -- REPLAY MODE LOOP (ANIMATED CHRONOLOGICAL STROKE PLAYBACK) --
            if replayer.is_replaying:
                prog = replayer.update_frame(canvas, dt)
                display_frame = get_background(frame, curr_bg)
                ink_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                _, ink_mask = cv2.threshold(ink_gray, 5, 255, cv2.THRESH_BINARY)
                if curr_bg == "WHITEBOARD":
                    display_frame[ink_mask > 0] = (30, 30, 35)
                else:
                    display_frame[ink_mask > 0] = canvas[ink_mask > 0]

                replayer.draw_hud(display_frame, prog)

                if vcam_mgr.is_active:
                    vcam_mgr.send_frame(display_frame)

                window_frame = cv2.resize(display_frame, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_AREA)
                cv2.imshow("Smart Air Writer v6.0 PRO", window_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord(' '):
                    replayer.toggle_pause()
                elif key == ord('s') or key == ord('S'):
                    new_spd = replayer.cycle_speed()
                    show_toast(f"Replay Speed: {new_spd:.0f}x", (0, 240, 255), 1.0)
                elif key == ord('y') or key == ord('Y'):
                    replayer.stop_replay()
                    canvas.fill(0)
                    doc.render(canvas)
                    show_toast("REPLAY EXITED", (200, 200, 200), 1.5)
                continue

            hands_pts = tracker.process(frame)

            # -- 1. COMPOSITING BACKGROUND & INK --
            display_frame = get_background(frame, curr_bg)

            ink_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, ink_mask = cv2.threshold(ink_gray, 5, 255, cv2.THRESH_BINARY)

            if curr_brush == "NEON":
                glow = cv2.GaussianBlur(canvas, (25, 25), 0)
                glow_mask = cv2.cvtColor(glow, cv2.COLOR_BGR2GRAY)
                _, glow_mask = cv2.threshold(glow_mask, 2, 255, cv2.THRESH_BINARY)
                glow_blend = cv2.addWeighted(display_frame, 1.0, glow, 0.75, 0)
                display_frame[glow_mask > 0] = glow_blend[glow_mask > 0]

            if curr_bg == "WHITEBOARD":
                display_frame[ink_mask > 0] = (30, 30, 35)
            else:
                display_frame[ink_mask > 0] = canvas[ink_mask > 0]

            # -- 2. MULTI-HAND ANALYSIS (PRO BIMANUAL SPATIAL ENGINE) --
            primary_gesture = "IDLE"
            primary_pts = None
            dominant_cursor = None
            open_palm_center = None

            hand_analyses = []
            for h_idx in range(min(2, len(hands_pts))):
                pts = hands_pts[h_idx]
                h_state = hand_states[h_idx]
                tracker.draw_skeleton(display_frame, pts, color=(140, 100, 200) if h_idx == 1 else (100, 140, 100))

                raw_gesture, conf, metrics = HandPoseEstimator.classify_hand_pro(pts)
                fsm_state, fsm_dur = gesture_fsms[h_idx].update(raw_gesture, metrics["norm_pinch"], now)

                # 1€ Filtered Tip (Zero jitter, Zero lag)
                ix, iy = pts[8]
                cx, cy = one_euro_hands[h_idx].filter(ix, iy, now)

                if fsm_state == "OPEN_PALM":
                    open_palm_center = tracker.get_palm_center(pts)
                else:
                    dominant_cursor = (cx, cy)

                hand_analyses.append({
                    "idx": h_idx,
                    "pts": pts,
                    "raw_gesture": raw_gesture,
                    "fsm_state": fsm_state,
                    "fsm_dur": fsm_dur,
                    "norm_pinch": metrics["norm_pinch"],
                    "cx": cx,
                    "cy": cy,
                    "h_state": h_state
                })

            if hand_analyses:
                primary_pts = hand_analyses[0]["pts"]
                primary_gesture = hand_analyses[0]["fsm_state"]
            else:
                one_euro_hands[0].reset()
                one_euro_hands[1].reset()
                palm_hold_start = None
                commit_hold_start = None

            # -- 3. HOLOGRAPHIC RADIAL MENU (Left Palm) --
            radial_action = None
            if open_palm_center is not None and not eraser_mode:
                radial_action = radial_menu.update_and_render(display_frame, open_palm_center, True, dominant_cursor, now)
            else:
                radial_menu.update_and_render(display_frame, (0, 0), False, None, now)

            if radial_action:
                act_type = radial_action["type"]
                act_val = radial_action["value"]
                if act_type == "COLOR":
                    CURRENT_COLOR_KEY = act_val
                    show_toast(f"COLOR: {PALETTE[CURRENT_COLOR_KEY][0]}", PALETTE[CURRENT_COLOR_KEY][1], 1.2)
                elif act_type == "BRUSH":
                    CURRENT_BRUSH_IDX = (CURRENT_BRUSH_IDX + 1) % len(BRUSH_TYPES)
                    show_toast(f"BRUSH: {BRUSH_TYPES[CURRENT_BRUSH_IDX]}", (0, 240, 255), 1.2)
                elif act_type == "ACTION":
                    if act_val == "TOGGLE_BG":
                        CURRENT_BG_IDX = (CURRENT_BG_IDX + 1) % len(BG_MODES)
                        show_toast(f"BOARD: {BG_MODES[CURRENT_BG_IDX]}", (255, 200, 0), 1.2)
                    elif act_val == "TOGGLE_ERASER":
                        eraser_mode = not eraser_mode
                        show_toast("ERASER ON" if eraser_mode else "DRAWING MODE", (0, 100, 255) if eraser_mode else (0, 255, 128), 1.2)
                    elif act_val == "UNDO":
                        res = doc.undo()
                        if res:
                            doc.render(canvas)
                            update_live_ocr()
                            show_toast("UNDO", (0, 220, 255), 1.2)
                    elif act_val == "CLEAR":
                        doc.clear_all()
                        canvas.fill(0)
                        committed_text = ""
                        live_word_text = ""
                        update_live_ocr()
                        show_toast("CANVAS CLEARED", (0, 100, 255), 1.2)

            # -- 4. SPATIAL WRITING & WORD MANIPULATION --
            for ha in hand_analyses:
                fsm_state = ha["fsm_state"]
                cx, cy = ha["cx"], ha["cy"]
                h_state = ha["h_state"]
                h_idx = ha["idx"]
                draw_color = PALETTE[CURRENT_COLOR_KEY][1] if h_idx == 0 else HAND_2_COLOR

                # Hand is in writing state if pinching (instant) OR index held for 0.8s
                is_pinching = (fsm_state == "PINCH_WRITE")
                is_index_pointing = (fsm_state == "INDEX_POINT")
                is_fist = (fsm_state == "FIST")

                # ✊ POING : stopper l'écriture immédiatement avec retour visuel
                if is_fist:
                    h_state["index_hold_start"] = None   # réinitialise le délai
                    if h_state["stroke"]:
                        finalize_hand_stroke(h_state, draw_color, curr_brush)
                    h_state["active"] = False
                    # Indicateur visuel ✊ poing
                    cv2.circle(display_frame, (cx, cy), 20, (60, 60, 200), 2)
                    cv2.putText(display_frame, "\u271a " + lang_mgr.get("PEN_UP"), (cx - 28, cy - 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 255), 1)
                    continue   # skip le reste pour cette main

                # -- Dwell timer : index doit être tenu 0.8s avant de commencer à écrire --
                if is_index_pointing:
                    if h_state["index_hold_start"] is None:
                        h_state["index_hold_start"] = now   # début du chrono
                    index_held_dur = now - h_state["index_hold_start"]
                    index_activated = (index_held_dur >= ACTIVATION_DELAY_SEC)

                    # Afficher le cercle de progression pendant l'attente
                    if not index_activated:
                        prog = min(1.0, index_held_dur / ACTIVATION_DELAY_SEC)
                        cv2.circle(display_frame, (cx, cy), 16, (0, 200, 255), 1)
                        cv2.ellipse(display_frame, (cx, cy), (16, 16), -90,
                                    0, int(prog * 360), (0, 255, 180), 3)
                    else:
                        # Activé : point plein draw_color
                        cv2.circle(display_frame, (cx, cy), PEN_THICKNESS + 4, draw_color, -1)
                        cv2.circle(display_frame, (cx, cy), PEN_THICKNESS + 7, (255, 255, 255), 2)
                else:
                    h_state["index_hold_start"] = None   # remise à zéro si doigt abaissé
                    index_activated = False

                # Écrire si : pincement (instantané) OU index activé après délai 0.8s
                is_writing = is_pinching or (is_index_pointing and index_activated)

                # A. Check Word Grab (Pinch only)
                word_grabbed = False
                if is_pinching:
                    m_stat, m_word = word_manipulator.check_grab(doc.committed_words, (cx, cy), is_pinching=True)
                    if m_stat == "GRABBED":
                        show_toast(f'GRABBED: "{m_word.text}"', (0, 240, 255), 1.0)
                        word_grabbed = True
                    elif m_stat == "DRAGGING":
                        doc.render(canvas)
                        word_manipulator.draw_highlight(display_frame, m_word)
                        word_grabbed = True

                # B. Direct Ink Writing (Pinch = instant, Index = après délai 0.8s)
                if is_writing and not word_grabbed:
                    if eraser_mode:
                        cv2.circle(display_frame, (cx, cy), 30, (0, 0, 255), 2)
                        if doc.erase_at((cx, cy), 30) > 0:
                            doc.render(canvas)
                            update_live_ocr()
                    else:
                        h_state["active"] = True
                        h_state["stroke"].append((cx, cy, now))
                        if is_pinching:
                            # Pinch : indicateur distinct (cercle + ring blanc)
                            cv2.circle(display_frame, (cx, cy), PEN_THICKNESS + 4, draw_color, -1)
                            cv2.circle(display_frame, (cx, cy), PEN_THICKNESS + 7, (255, 255, 255), 2)
                        if len(h_state["stroke"]) > 1:
                            render_stroke_smooth(canvas, h_state["stroke"][-4:], draw_color, PEN_THICKNESS, curr_brush)
                else:
                    # Pen Lift / Released (Finger curled, 2-fingers hover, or word drop)
                    m_stat, m_word = word_manipulator.check_grab(doc.committed_words, (cx, cy), is_pinching=False)
                    if m_stat == "DELETE":
                        if m_word in doc.committed_words:
                            doc.committed_words.remove(m_word)
                        doc.render(canvas)
                        show_toast(f'DELETED: "{m_word.text}"', (0, 100, 255), 1.5)
                    elif h_state["stroke"]:
                        finalize_hand_stroke(h_state, draw_color, curr_brush)
                    h_state["active"] = False

                    # Two fingers = Hover cursor (Pen Up / move without drawing)
                    if fsm_state == "TWO_FINGERS_HOVER":
                        cv2.circle(display_frame, (cx, cy), 9, (0, 255, 255), 2)
                        cv2.circle(display_frame, (cx, cy), 3, (0, 255, 255), -1)
                        cv2.putText(display_frame, "HOVER", (cx + 12, cy - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
                    elif fsm_state in ("IDLE", "FIST"):
                        # Subtle small point to show tracking
                        cv2.circle(display_frame, (cx, cy), 3, (140, 140, 140), 1)

            # -- 5. SPONGE ERASER (Open Palm Held on Ink) --
            if primary_gesture == "OPEN_PALM" and primary_pts is not None and open_palm_center is not None:
                if palm_hold_start is None:
                    palm_hold_start = now
                p_dur = now - palm_hold_start
                p_prog = min(1.0, p_dur / PALM_HOLD_SEC)
                bar_col = (0, 0, 255) if p_prog >= 1.0 else (0, 140, 255)

                cv2.circle(display_frame, open_palm_center, ERASER_RADIUS, (0, 0, 255) if p_prog >= 1.0 else (100, 100, 100), 2)
                cv2.ellipse(display_frame, open_palm_center, (ERASER_RADIUS, ERASER_RADIUS),
                            -90, 0, int(p_prog * 360), bar_col, 3)

                if p_prog >= 1.0:
                    if doc.erase_at(open_palm_center, ERASER_RADIUS) > 0:
                        doc.render(canvas)
                        update_live_ocr()
            else:
                palm_hold_start = None

            # -- 6. THUMBS UP / OK COMMIT PROGRESS RING --
            if primary_gesture == "THUMB_UP" and primary_pts is not None:
                if now >= commit_cooldown_until:
                    if commit_hold_start is None:
                        commit_hold_start = now
                    c_dur = now - commit_hold_start
                    c_prog = min(1.0, c_dur / COMMIT_HOLD_SEC)
                    thumb_pt = primary_pts[4]

                    cv2.circle(display_frame, thumb_pt, 28, (80, 80, 80), 2)
                    cv2.ellipse(display_frame, thumb_pt, (28, 28), -90, 0, int(c_prog * 360), (0, 255, 128), 3)
                    cv2.putText(display_frame, "OK", (thumb_pt[0] - 14, thumb_pt[1] + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 128), 2)

                    if c_prog >= 1.0:
                        if live_word_text:
                            corrector.learn(live_word_text)
                            assistant.add_word(live_word_text)
                            doc.commit_current_word(live_word_text)
                            committed_text += live_word_text + " "
                            show_toast(f'{lang_mgr.get("COMMITTED")}: "{live_word_text}" [OK]', (0, 255, 128), 2.0)
                        else:
                            committed_text += " "
                            show_toast(f'{lang_mgr.get("SPACE")} [OK]', (0, 255, 128), 1.5)
                        live_word_text = ""
                        live_word_alts = []
                        current_best_raw = ""
                        alt_idx = 0
                        commit_hold_start = None
                        commit_cooldown_until = now + 1.2
                else:
                    commit_hold_start = None
            else:
                commit_hold_start = None

            # -- 1. TOP BAR HUD --
            cv2.rectangle(display_frame, (0, 0), (FRAME_WIDTH, 60), (18, 18, 22), -1)
            cv2.line(display_frame, (0, 60), (FRAME_WIDTH, 60), (55, 55, 65), 1)

            cv2.putText(display_frame, lang_mgr.get("TITLE"), (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2)
            if eraser_mode:
                cv2.putText(display_frame, f"MODE: {lang_mgr.get('ERASER')} [E]", (290, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 100, 255), 2)
            else:
                cv2.putText(display_frame, f"{lang_mgr.get('BRUSH')}: {curr_brush} [B]", (290, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1)

            # Auto-Shape Badge
            shape_txt = "Shape: ON [F]" if auto_shapes_enabled else "Shape: OFF [F]"
            shape_col = (0, 255, 180) if auto_shapes_enabled else (120, 120, 120)
            cv2.putText(display_frame, shape_txt, (475, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, shape_col, 1)

            # Virtual Camera Badge
            if vcam_mgr.is_active:
                cv2.circle(display_frame, (625, 33), 5, (0, 255, 128), -1)
                cv2.putText(display_frame, "V-Cam: ON [U]", (635, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 128), 1)
            else:
                cv2.putText(display_frame, "V-Cam [U]", (625, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (130, 130, 130), 1)

            # Word count
            word_count = len(doc.committed_words) + (1 if not doc.current_word.is_empty() else 0)
            cv2.putText(display_frame, f"{lang_mgr.get('WORDS')}: {word_count}", (735, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 128), 1)

            # Recording badge
            if is_recording:
                rec_secs = int(now - record_start_time)
                if int(now * 2) % 2 == 0:
                    cv2.circle(display_frame, (830, 33), 6, (0, 0, 255), -1)
                cv2.putText(display_frame, f"REC {rec_secs // 60:02d}:{rec_secs % 60:02d}", (842, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "Rec [V]", (830, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 140, 140), 1)

            # Replay badge
            cv2.putText(display_frame, "Replay [Y]", (905, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 200, 255), 1)

            # Notes Export badge
            cv2.putText(display_frame, "Notes [M]", (995, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 180, 0), 1)

            # Language badge
            cv2.putText(display_frame, f"[{lang_mgr.lang}]", (1080, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 200), 2)

            # Palette color circle
            col_name, col_bgr = PALETTE[CURRENT_COLOR_KEY]
            cv2.circle(display_frame, (FRAME_WIDTH - 65, 33), 11, col_bgr, -1)
            cv2.circle(display_frame, (FRAME_WIDTH - 65, 33), 13, (200, 200, 200), 2)

            # -- 2. AUTO-UPDATING TEXT BAR (just under the top bar) --
            cv2.rectangle(display_frame, (0, 62), (FRAME_WIDTH, 110), (16, 18, 26), -1)
            cv2.line(display_frame, (0, 110), (FRAME_WIDTH, 110), (50, 55, 70), 1)

            cursor_blink = "|" if int(now * 2) % 2 == 0 else " "
            display_text = (committed_text + live_word_text)[-35:] + cursor_blink

            cv2.putText(display_frame, "TEXT: ", (20, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 240, 255), 2)
            cv2.putText(display_frame, display_text, (95, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

            # Alternatives suggestions indicator
            if live_word_alts:
                alts_str = " | ".join(live_word_alts[:3])
                cv2.putText(display_frame, f"[Tab: {alts_str}]", (540, 96),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 255), 1)

            # OCR engine badge (right side of the text bar)
            if ocr.backend_name in ("EasyOCR", "EMNIST-CNN"):
                ocr_badge_col = (0, 255, 128)     # green: best
            elif ocr.backend_name in ("EMNIST-MLP", "SYNTH-NET"):
                ocr_badge_col = (0, 240, 255)     # cyan: good
            else:
                ocr_badge_col = (0, 140, 255)     # orange: basic fallback

            if ocr.mode == "auto":
                num_label = "[N: AUTO (Words & 0-9)]"
                num_col = (0, 255, 128)
            elif ocr.mode == "digits":
                num_label = "[N: DIGITS ONLY]"
                num_col = (0, 220, 255)
            else:
                num_label = "[N: LETTERS ONLY]"
                num_col = (200, 150, 255)

            cv2.putText(display_frame, num_label, (FRAME_WIDTH - 390, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, num_col, 1)
            cv2.putText(display_frame, f"OCR: {ocr.backend_name}", (FRAME_WIDTH - 220, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, ocr_badge_col, 2)

            # -- 3. TOAST (floats below the text bar) --
            if now < toast_exp:
                tw = cv2.getTextSize(toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)[0][0]
                tx = (FRAME_WIDTH - tw) // 2
                cv2.rectangle(display_frame, (tx - 15, 118), (tx + tw + 15, 158), (20, 20, 25), -1)
                cv2.rectangle(display_frame, (tx - 15, 118), (tx + tw + 15, 158), toast_col, 2)
                cv2.putText(display_frame, toast_msg, (tx, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, toast_col, 2)

            # -- 3b. CALIBRATION BANNER (if active) --
            if calibration_mgr.is_active:
                cv2.rectangle(display_frame, (0, 112), (FRAME_WIDTH, 155), (18, 26, 38), -1)
                cv2.line(display_frame, (0, 155), (FRAME_WIDTH, 155), (0, 255, 180), 2)
                cal_str = f"CALIBRATION: Trace '{calibration_mgr.current_char}' in air ({calibration_mgr.char_index+1}/{len(calibration_mgr.CALIBRATION_CHARS)}) [Pinch to write, [K] to exit]"
                cv2.putText(display_frame, cal_str, (30, 142),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)

            # -- 4. FOOTER (plain text) --
            cv2.rectangle(display_frame, (0, FRAME_HEIGHT - 30), (FRAME_WIDTH, FRAME_HEIGHT), (15, 15, 20), -1)
            cv2.line(display_frame, (0, FRAME_HEIGHT - 30), (FRAME_WIDTH, FRAME_HEIGHT - 30), (45, 45, 55), 1)
            footer = "Pinch: Write (0s) | Palm: Radial Tools | Thumb Up: Commit [OK] | Pinch Word: Move/Trash | [K] Calib | [Tab] Alts | [N] Mode | [Space] Commit | [Z] Undo | [C] Clear"
            cv2.putText(display_frame, footer, (20, FRAME_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

            # Recording stays in full 1280x720
            if is_recording and video_writer is not None:
                video_writer.write(display_frame)

            # Virtual Camera Stream (Zoom / Teams / Meet)
            if vcam_mgr.is_active:
                vcam_mgr.send_frame(display_frame)

            # Window resized to 1024x576 for small screens
            window_frame = cv2.resize(display_frame, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_AREA)
            cv2.imshow("Smart Air Writer v6.0 PRO", window_frame)

            # -- KEYS --
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('k') or key == ord('K'):
                if calibration_mgr.is_active:
                    calibration_mgr.stop()
                    show_toast("CALIBRATION STOPPED", (140, 140, 140), 1.8)
                else:
                    calibration_mgr.start()
                    show_toast(f"CALIBRATION: Trace '{calibration_mgr.current_char}'", (0, 255, 180), 2.5)
            elif key == ord('n') or key == ord('N'):
                mode_cycle = {"auto": "digits", "digits": "letters", "letters": "auto"}
                ocr.mode = mode_cycle.get(ocr.mode, "auto")
                if ocr.mode == "auto":
                    show_toast("MODE: AUTO (Words & Digits)", (0, 255, 128), 2.0)
                elif ocr.mode == "digits":
                    show_toast("MODE: DIGITS ONLY [0-9]", (0, 220, 255), 2.0)
                else:
                    show_toast("MODE: LETTERS ONLY [A-Z]", (200, 150, 255), 2.0)
                update_live_ocr()
            elif key == 9:  # [Tab] key: cycle alternatives
                if live_word_alts:
                    alt_idx = (alt_idx + 1) % (len(live_word_alts) + 1)
                    if alt_idx == 0:
                        live_word_text = current_best_raw
                    else:
                        live_word_text = live_word_alts[alt_idx - 1]
                    show_toast(f'Alt ({alt_idx + 1}/{len(live_word_alts) + 1}): "{live_word_text}"', (0, 240, 255), 2.0)
                else:
                    show_toast("No alternatives", (140, 140, 140), 1.2)
            elif key in (8, 127):  # [Backspace] key: delete character
                if live_word_text:
                    live_word_text = live_word_text[:-1]
                    current_best_raw = live_word_text
                    show_toast(f'Del: "{live_word_text}"', (0, 200, 255), 1.0)
                elif committed_text:
                    committed_text = committed_text.rstrip()
                    committed_text = committed_text[:-1] if committed_text else ""
                    show_toast("Del char", (0, 200, 255), 1.0)
            elif key == ord('e') or key == ord('E'):
                eraser_mode = not eraser_mode
                if eraser_mode:
                    show_toast("ERASER ON [finger erases]", (0, 100, 255), 2.2)
                else:
                    show_toast("DRAWING MODE [writing ink]", (0, 255, 128), 2.2)
            elif key == ord('z') or key == ord('Z'):
                undo_res = doc.undo()
                if undo_res:
                    doc.render(canvas)
                    update_live_ocr()
                    show_toast("UNDO", (0, 220, 255), 1.8)
                else:
                    show_toast("Nothing to undo", (140, 140, 140), 1.5)
            elif key == ord(' '):
                # Space commits the current word, learns it, and moves to the next
                if live_word_text:
                    corrector.learn(live_word_text)
                    assistant.add_word(live_word_text)
                    doc.commit_current_word(live_word_text)
                    committed_text += live_word_text + " "
                    show_toast(f'{lang_mgr.get("COMMITTED")}: "{live_word_text}"', (0, 240, 255), 1.6)
                else:
                    committed_text += " "
                    show_toast(lang_mgr.get("SPACE"), (0, 240, 255), 1.2)
                live_word_text = ""
                live_word_alts = []
                current_best_raw = ""
                alt_idx = 0
            elif key == ord('f') or key == ord('F'):
                auto_shapes_enabled = not auto_shapes_enabled
                status_txt = "ON" if auto_shapes_enabled else "OFF"
                col = (0, 255, 180) if auto_shapes_enabled else (140, 140, 140)
                show_toast(f"AUTO-SHAPE SNAP: {status_txt}", col, 2.0)
            elif key == ord('y') or key == ord('Y'):
                if replayer.all_strokes:
                    canvas.fill(0)
                    replayer.start_replay()
                    show_toast("SESSION REPLAY STARTED [Space: Pause, S: Speed, Y: Exit]", (0, 240, 255), 2.5)
                else:
                    show_toast("No strokes to replay yet! Draw something first.", (100, 100, 255), 2.0)
            elif key == ord('m') or key == ord('M'):
                notes_path = assistant.export_markdown()
                notes_fn = os.path.basename(notes_path)
                show_toast(f"SAVED: exports/{notes_fn}", (0, 255, 128), 3.0)
            elif key == ord('u') or key == ord('U'):
                ok, msg = vcam_mgr.toggle()
                col = (0, 255, 128) if vcam_mgr.is_active else (0, 140, 255)
                show_toast(msg, col, 2.5)
            elif key == ord('l') or key == ord('L'):
                new_lang, lang_lbl = lang_mgr.cycle_language()
                show_toast(lang_lbl, (0, 240, 255), 2.0)
            elif key == ord('r') or key == ord('R'):
                update_live_ocr()
                if live_word_text:
                    show_toast(f'OCR: "{live_word_text}"', (0, 255, 128), 2.0)
                else:
                    show_toast("No word detected", (140, 140, 140), 1.5)
            elif key == ord('b') or key == ord('B'):
                CURRENT_BRUSH_IDX = (CURRENT_BRUSH_IDX + 1) % len(BRUSH_TYPES)
                show_toast(f"Brush: {BRUSH_TYPES[CURRENT_BRUSH_IDX]}", (0, 240, 255))
            elif key == ord('w') or key == ord('W'):
                CURRENT_BG_IDX = (CURRENT_BG_IDX + 1) % len(BG_MODES)
                show_toast(f"Background: {BG_MODES[CURRENT_BG_IDX]}", (255, 200, 0))
            elif key == ord('v') or key == ord('V'):
                if not is_recording:
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_path = os.path.join(EXPORTS_DIR, f"air_session_{ts_str}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(rec_path, fourcc, 25.0, (FRAME_WIDTH, FRAME_HEIGHT))
                    is_recording = True
                    record_start_time = now
                    show_toast("Recording started (1280x720)", (0, 0, 255), 2.5)
                else:
                    is_recording = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    show_toast("Video saved in exports/", (0, 255, 128), 2.5)
            elif key == ord('p') or key == ord('P'):
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                png_path = os.path.join(EXPORTS_DIR, f"whiteboard_{ts_str}.png")
                svg_path = os.path.join(EXPORTS_DIR, f"whiteboard_{ts_str}.svg")
                cv2.imwrite(png_path, display_frame)
                doc.export_svg(svg_path, FRAME_WIDTH, FRAME_HEIGHT)
                show_toast("Exported PNG & SVG in exports/", (0, 255, 128), 3.0)
            elif key == ord('c') or key == ord('C'):
                doc.clear_all()
                canvas.fill(0)
                committed_text = ""
                live_word_text = ""
                live_word_alts = []
                current_best_raw = ""
                alt_idx = 0
                show_toast("CANVAS & TEXT CLEARED", (0, 100, 255))
            elif key != 255 and chr(key) in PALETTE:
                CURRENT_COLOR_KEY = chr(key)
            elif key != 255 and chr(key) in PALETTE:
                CURRENT_COLOR_KEY = chr(key)

    finally:
        if vcam_mgr:
            vcam_mgr.stop()
        if video_writer:
            video_writer.release()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Application terminated.")


if __name__ == "__main__":
    main()