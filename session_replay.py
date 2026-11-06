"""
SESSION ANIMATED REPLAYER v6.0
Live chronological stroke animation playback engine for Smart Air Writer.

Replays all session strokes point-by-point based on captured timestamps:
- Variable speeds: 1x, 2x, 4x, 8x
- Interactive Play / Pause / Seek
- Playback HUD overlay
"""

import time
import numpy as np
import cv2
from spline_renderer import render_stroke_smooth


class ReplayStroke:
    def __init__(self, points, color, thickness, brush):
        self.points = list(points)  # [(x, y, t), ...]
        self.color = color
        self.thickness = thickness
        self.brush = brush


class SessionReplayer:
    def __init__(self):
        self.all_strokes = []
        self.is_replaying = False
        self.is_paused = False
        self.speed = 2.0
        self.current_idx = 0
        self.total_points = 0
        self.flat_points = []  # [(stroke_id, pt_idx, x, y, t, color, thick, brush)]
        self.start_replay_time = 0.0
        self.replay_sim_time = 0.0

    def record_stroke(self, stroke_item):
        """Records a finalized stroke into replay timeline."""
        if not stroke_item or len(stroke_item.points) < 2:
            return
        self.all_strokes.append(
            ReplayStroke(stroke_item.points, stroke_item.color, stroke_item.thickness, stroke_item.brush)
        )

    def start_replay(self):
        """Prepares flattened timeline and starts playback."""
        if not self.all_strokes:
            return False

        self.flat_points = []
        t0 = None
        for s_id, s in enumerate(self.all_strokes):
            for p_id, p in enumerate(s.points):
                t = p[2] if len(p) > 2 else (t0 or 0.0)
                if t0 is None:
                    t0 = t
                rel_t = max(0.0, t - t0)
                self.flat_points.append({
                    "s_id": s_id,
                    "p_id": p_id,
                    "x": int(p[0]),
                    "y": int(p[1]),
                    "rel_t": rel_t,
                    "color": s.color,
                    "thick": s.thickness,
                    "brush": s.brush,
                    "is_start": (p_id == 0)
                })

        self.total_points = len(self.flat_points)
        self.current_idx = 0
        self.is_replaying = True
        self.is_paused = False
        self.start_replay_time = time.time()
        self.replay_sim_time = 0.0
        return True

    def stop_replay(self):
        self.is_replaying = False
        self.is_paused = False

    def toggle_pause(self):
        self.is_paused = not self.is_paused

    def cycle_speed(self):
        speeds = [1.0, 2.0, 4.0, 8.0]
        curr_i = speeds.index(self.speed) if self.speed in speeds else 0
        self.speed = speeds[(curr_i + 1) % len(speeds)]
        return self.speed

    def update_frame(self, canvas, dt):
        """
        Advances animation based on delta time and renders accumulated strokes onto canvas.
        Returns: progress_ratio (0.0 to 1.0)
        """
        if not self.is_replaying or not self.flat_points:
            return 1.0

        if not self.is_paused:
            self.replay_sim_time += dt * self.speed
            # Advance index to match sim time
            while (self.current_idx < self.total_points and
                   self.flat_points[self.current_idx]["rel_t"] <= self.replay_sim_time):
                item = self.flat_points[self.current_idx]
                s_id = item["s_id"]
                p_id = item["p_id"]

                # Draw recent segment
                if p_id > 0:
                    prev_item = self.flat_points[self.current_idx - 1]
                    if prev_item["s_id"] == s_id:
                        pts = [(prev_item["x"], prev_item["y"]), (item["x"], item["y"])]
                        render_stroke_smooth(canvas, pts, item["color"], item["thick"], item["brush"])

                self.current_idx += 1

        prog = min(1.0, self.current_idx / max(1, self.total_points))
        if self.current_idx >= self.total_points:
            # Finished
            self.is_paused = True

        return prog

    def draw_hud(self, frame, progress):
        """Draws interactive replay controls overlay."""
        h, w = frame.shape[:2]
        bar_y = h - 60
        # Bottom controls bar
        cv2.rectangle(frame, (0, bar_y), (w, h), (18, 18, 24), -1)
        cv2.line(frame, (0, bar_y), (w, bar_y), (50, 50, 65), 1)

        # Progress bar
        bar_x1, bar_x2 = 220, w - 240
        cv2.line(frame, (bar_x1, bar_y + 30), (bar_x2, bar_y + 30), (45, 45, 55), 4)
        prog_x = int(bar_x1 + progress * (bar_x2 - bar_x1))
        cv2.line(frame, (bar_x1, bar_y + 30), (prog_x, bar_y + 30), (0, 240, 255), 4)
        cv2.circle(frame, (prog_x, bar_y + 30), 7, (0, 240, 255), -1)

        # Status labels
        status_txt = "PAUSED [Space]" if self.is_paused else "PLAYING [Space]"
        cv2.putText(frame, f"REPLAY: {status_txt}", (25, bar_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 180) if not self.is_paused else (0, 200, 255), 1)

        # Speed and percentage
        pct = int(progress * 100)
        cv2.putText(frame, f"{pct}%", (bar_x2 + 15, bar_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        cv2.putText(frame, f"Speed: {self.speed:.0f}x [S]", (bar_x2 + 75, bar_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1)
        cv2.putText(frame, "Exit [Y]", (w - 75, bar_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 255), 1)
