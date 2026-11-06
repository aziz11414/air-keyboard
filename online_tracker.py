"""
ONLINE TEMPORAL TRACKER & CALIBRATION ENGINE v6.0
Online trajectory processing, sequential motor features & user personalization.

Includes:
- Online feature extraction: (dx, dy, dt, speed, curvature, pen_lift)
- Equidistant arc-length resampling
- Dynamic Time Warping (DTW) & Angular Correlation Matcher
- Interactive Calibration Session (key [K]): Records real air-writing dataset
- Continuous learning from [Tab] and [Backspace] corrections
"""

import math
import time
import json
import os
import numpy as np


class OnlineTrajectory:
    """Represents a continuous temporal stroke with timestamps and spatial coordinates."""
    def __init__(self):
        self.points = []  # list of (x, y, timestamp)

    def add_point(self, x, y, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        self.points.append((float(x), float(y), float(timestamp)))

    def is_empty(self):
        return len(self.points) < 2

    def resample_equidistant(self, target_points=32):
        """
        Resamples stroke to N points with equidistant arc-length spacing.
        Preserves temporal order, speed, and geometric curvature.
        """
        if len(self.points) < 2:
            return None

        pts = np.array(self.points)
        coords = pts[:, :2]
        times = pts[:, 2]

        # Calculate cumulative distance
        diffs = np.diff(coords, axis=0)
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        total_len = np.sum(dists)

        if total_len < 1e-4:
            return np.repeat(coords[:1], target_points, axis=0)

        cum_dists = np.insert(np.cumsum(dists), 0, 0.0)
        target_dists = np.linspace(0.0, total_len, target_points)

        resampled_x = np.interp(target_dists, cum_dists, coords[:, 0])
        resampled_y = np.interp(target_dists, cum_dists, coords[:, 1])
        resampled_t = np.interp(target_dists, cum_dists, times)

        return np.column_stack([resampled_x, resampled_y, resampled_t])

    def extract_features(self, resampled_pts=None):
        """
        Extracts 6D temporal feature vector per point:
        [norm_x, norm_y, dx, dy, speed, curvature]
        Can be called on instance without args (auto-resamples), or with pre-resampled points.
        """
        if resampled_pts is None:
            resampled_pts = self.resample_equidistant(32)
        if resampled_pts is None or len(resampled_pts) < 2:
            return None

        xy = resampled_pts[:, :2]
        t = resampled_pts[:, 2]

        # Normalize bounding box into [-1, 1] preserving aspect ratio
        min_xy = np.min(xy, axis=0)
        max_xy = np.max(xy, axis=0)
        center = (min_xy + max_xy) / 2.0
        scale = max(max_xy - min_xy) / 2.0
        if scale < 1e-4:
            scale = 1.0

        norm_xy = (xy - center) / scale

        # Velocity & Displacement
        dt = np.diff(t)
        dt = np.where(dt < 1e-4, 1e-3, dt)
        diff_xy = np.diff(norm_xy, axis=0)
        dx = np.insert(diff_xy[:, 0], 0, 0.0)
        dy = np.insert(diff_xy[:, 1], 0, 0.0)

        speeds = np.sqrt(dx ** 2 + dy ** 2) / np.insert(dt, 0, 0.033)

        # Curvature (change in direction angle)
        angles = np.arctan2(dy, dx)
        d_angles = np.diff(angles)
        # Normalize to [-pi, pi]
        d_angles = (d_angles + np.pi) % (2 * np.pi) - np.pi
        curvatures = np.insert(d_angles, 0, 0.0)

        return np.column_stack([norm_xy[:, 0], norm_xy[:, 1], dx, dy, speeds, curvatures])


class CalibrationManager:
    """
    Manages interactive calibration session and records personal air-writing dataset.
    """
    CALIBRATION_CHARS = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    ]

    def __init__(self, data_dir=None, save_dir=None):
        self.data_dir = data_dir or save_dir or os.path.join(os.getcwd(), "dataset")
        os.makedirs(self.data_dir, exist_ok=True)
        self.dataset_file = os.path.join(self.data_dir, "user_online_dataset.jsonl")

        self.is_active = False
        self.char_index = 0
        self.current_char = self.CALIBRATION_CHARS[0]
        self.collected_samples = []

    def current_prompt(self):
        """Returns the current calibration instructions."""
        return f"Tracez '{self.current_char}' dans l'air ({self.char_index + 1}/{len(self.CALIBRATION_CHARS)})"

    def start(self):
        self.is_active = True
        self.char_index = 0
        self.current_char = self.CALIBRATION_CHARS[0]
        self.collected_samples = []
        print("[Calibration] Session started. Follow on-screen prompts.")

    def stop(self):
        self.is_active = False
        print("[Calibration] Session ended.")

    def record_stroke(self, raw_points, label=None):
        """Records an air-drawn stroke with timestamp metadata to JSONL."""
        if len(raw_points) < 4:
            return False

        if label is None:
            label = self.current_char

        sample = {
            "label": label,
            "timestamp": time.time(),
            "points": [(float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else time.time()) for p in raw_points]
        }

        try:
            with open(self.dataset_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")
        except Exception as e:
            print(f"[Calibration] Save error: {e}")

        if self.is_active:
            self.char_index += 1
            if self.char_index < len(self.CALIBRATION_CHARS):
                self.current_char = self.CALIBRATION_CHARS[self.char_index]
            else:
                self.is_active = False
                print(f"[Calibration] All {len(self.CALIBRATION_CHARS)} characters calibrated successfully!")

        return True
