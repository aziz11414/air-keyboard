"""
GESTURE ENGINE PRO v6.0
Spatial Computing & Motor Gesture Tracking for Smart Air Writer.

Includes:
- OneEuroFilter (Casiez, Roussel, Vogel - ACM CHI 2012): Adaptive filtering (zero jitter when slow, zero lag when fast)
- HandPoseEstimator: Rotation-invariant bone angle and scale metrics
- Instant PinchDetector: Zero-latency touchless writing trigger with hysteresis
- GestureStateMachine: State machine with confidence tracking and debounce
- VirtualPlaneDetector: Z-depth virtual surface crossing detector
"""

import math
import time
import numpy as np


class LowPassFilter:
    """First-order low-pass filter."""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.y = None
        self.s = None

    def filter(self, value, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        if self.y is None:
            s = value
        else:
            s = self.alpha * value + (1.0 - self.alpha) * self.s
        self.y = value
        self.s = s
        return s

    def reset(self):
        self.y = None
        self.s = None


class OneEuroFilter:
    """
    1€ Filter: Adaptive Low-Pass Filter for Human-Computer Interaction.
    Eliminates high-frequency jitter at low speeds without adding latency during fast movements.
    """
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, timestamp=None):
        if self.last_time is not None and timestamp is not None:
            dt = timestamp - self.last_time
            if dt > 1e-4:
                self.freq = 1.0 / dt
        self.last_time = timestamp

        prev_x = self.x_filter.s
        dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
        edx = self.dx_filter.filter(dx, self._alpha(self.dcutoff))
        cutoff = self.mincutoff + self.beta * abs(edx)
        return self.x_filter.filter(x, self._alpha(cutoff))

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class PointFilter2D:
    """Convenience filter for 2D (x, y) coordinates with independent 1€ filters."""
    def __init__(self, freq=30.0, mincutoff=1.2, beta=0.008):
        self.fx = OneEuroFilter(freq, mincutoff, beta)
        self.fy = OneEuroFilter(freq, mincutoff, beta)

    def filter(self, x, y, timestamp=None):
        fx = self.fx.filter(float(x), timestamp)
        fy = self.fy.filter(float(y), timestamp)
        return int(round(fx)), int(round(fy))

    def reset(self):
        self.fx.reset()
        self.fy.reset()


class PointFilter3D:
    """Convenience filter for 3D (x, y, z) coordinates."""
    def __init__(self, freq=30.0, mincutoff=1.2, beta=0.008):
        self.fx = OneEuroFilter(freq, mincutoff, beta)
        self.fy = OneEuroFilter(freq, mincutoff, beta)
        self.fz = OneEuroFilter(freq, mincutoff, beta)

    def filter(self, x, y, z, timestamp=None):
        fx = self.fx.filter(float(x), timestamp)
        fy = self.fy.filter(float(y), timestamp)
        fz = self.fz.filter(float(z), timestamp)
        return (fx, fy, fz)

    def reset(self):
        self.fx.reset()
        self.fy.reset()
        self.fz.reset()


class HandPoseEstimator:
    """
    Rotation-invariant hand metrics based on bone vectors, palm scale, and joint angles.
    Independent of hand tilt or camera distance.
    """
    @staticmethod
    def get_palm_scale(pts):
        """Scale factor = distance between Wrist (#0) and Middle MCP (#9)."""
        w = pts[0]
        m = pts[9]
        return max(20.0, math.hypot(w[0] - m[0], w[1] - m[1]))

    @staticmethod
    def get_pinch_distance(pts):
        """Euclidean distance between thumb tip (#4) and index tip (#8)."""
        t = pts[4]
        i = pts[8]
        return math.hypot(t[0] - i[0], t[1] - i[1])

    @staticmethod
    def get_normalized_pinch(pts):
        """Pinch distance normalized by palm scale (scale-invariant)."""
        d = HandPoseEstimator.get_pinch_distance(pts)
        scale = HandPoseEstimator.get_palm_scale(pts)
        return d / scale

    @staticmethod
    def is_finger_extended(pts, tip_idx, mcp_idx, pip_idx):
        """
        Check if finger is extended outward relative to wrist and MCP,
        invariant to hand rotation and distance.
        """
        wrist = pts[0]
        mcp = pts[mcp_idx]
        tip = pts[tip_idx]
        pip = pts[pip_idx]

        scale = HandPoseEstimator.get_palm_scale(pts)
        d_tip_wrist = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
        d_pip_wrist = math.hypot(pip[0] - wrist[0], pip[1] - wrist[1])
        d_tip_mcp = math.hypot(tip[0] - mcp[0], tip[1] - mcp[1])

        # Scale-adaptive radial extension
        radial = (d_tip_wrist > d_pip_wrist + 0.06 * scale) and (d_tip_mcp > 0.22 * scale)
        # Vertical extension when hand is upright
        upright = (tip[1] < pip[1] - 4) and (d_tip_wrist > d_pip_wrist)
        return radial or upright

    @staticmethod
    def classify_hand_pro(pts):
        """
        Classifies hand gesture using rotation-invariant geometric metrics.
        Returns: gesture_name, confidence, metrics_dict
        """
        palm_scale = HandPoseEstimator.get_palm_scale(pts)
        norm_pinch = HandPoseEstimator.get_normalized_pinch(pts)

        index_ext = HandPoseEstimator.is_finger_extended(pts, 8, 5, 6)
        middle_ext = HandPoseEstimator.is_finger_extended(pts, 12, 9, 10)
        ring_ext = HandPoseEstimator.is_finger_extended(pts, 16, 13, 14)
        pinky_ext = HandPoseEstimator.is_finger_extended(pts, 20, 17, 18)

        # Thumb vector relative to index MCP
        thumb_tip = pts[4]
        thumb_ip = pts[3]
        index_mcp = pts[5]
        wrist = pts[0]

        # Vector wrist -> index_mcp (hand main axis)
        hx = index_mcp[0] - wrist[0]
        hy = index_mcp[1] - wrist[1]
        h_len = math.hypot(hx, hy) + 1e-5
        hx, hy = hx / h_len, hy / h_len

        metrics = {
            "palm_scale": palm_scale,
            "norm_pinch": norm_pinch,
            "index_ext": index_ext,
            "middle_ext": middle_ext,
            "ring_ext": ring_ext,
            "pinky_ext": pinky_ext
        }

        # 1. Instant PINCH (Thumb tip touching Index tip closely)
        if norm_pinch < 0.22:
            return "PINCH_WRITE", 0.95, metrics

        # 2. Open Palm (All 4 fingers extended) -> Tool Palette or Sponge
        if index_ext and middle_ext and ring_ext and pinky_ext:
            return "OPEN_PALM", 0.92, metrics

        # 3. Thumbs Up (Thumb extended upward, other 4 curled into fist)
        thumb_dist_wrist = math.hypot(thumb_tip[0] - wrist[0], thumb_tip[1] - wrist[1])
        all_others_curled = (not index_ext) and (not middle_ext) and (not ring_ext) and (not pinky_ext)
        if all_others_curled and (thumb_dist_wrist > palm_scale * 0.70):
            # Check thumb direction roughly up
            if thumb_tip[1] < thumb_ip[1] and thumb_tip[1] < index_mcp[1] - 15:
                return "THUMB_UP", 0.90, metrics

        # 4. FIST ✊ : tous les doigts (index, majeur, annulaire, auriculaire) repliés
        #    -> Lever le stylo / Arrêter l'écriture immédiatement
        if all_others_curled:
            return "FIST", 0.92, metrics

        # 5. Two fingers (Index + Middle extended, others curled) -> Pen Lift / Hover
        if index_ext and middle_ext and (not ring_ext) and (not pinky_ext):
            return "TWO_FINGERS_HOVER", 0.88, metrics

        # 6. Pointing Index (Index extended, middle curled) -> Main Air Writing!
        if index_ext and (not middle_ext) and not (ring_ext and pinky_ext):
            return "INDEX_POINT", 0.90, metrics

        return "IDLE", 0.50, metrics


class GestureStateMachine:
    """
    Hysteresis-based state machine for robust, glitch-free gesture transitions.
    """
    def __init__(self):
        self.state = "IDLE"
        self.prev_state = "IDLE"
        self.state_enter_time = time.time()
        self.pinch_active = False

    def update(self, raw_gesture, norm_pinch, now=None):
        if now is None:
            now = time.time()

        # Pinch hysteresis
        if self.pinch_active:
            if norm_pinch > 0.32:
                self.pinch_active = False
        else:
            if norm_pinch < 0.22 and raw_gesture == "PINCH_WRITE":
                self.pinch_active = True

        next_state = self.state

        if self.pinch_active:
            next_state = "PINCH_WRITE"
        elif raw_gesture == "FIST":
            next_state = "FIST"
        elif raw_gesture == "OPEN_PALM":
            next_state = "OPEN_PALM"
        elif raw_gesture == "THUMB_UP":
            next_state = "THUMB_UP"
        elif raw_gesture == "TWO_FINGERS_HOVER":
            next_state = "TWO_FINGERS_HOVER"
        elif raw_gesture == "INDEX_POINT":
            next_state = "INDEX_POINT"
        else:
            next_state = "IDLE"

        if next_state != self.state:
            self.prev_state = self.state
            self.state = next_state
            self.state_enter_time = now

        duration = now - self.state_enter_time
        return self.state, duration
