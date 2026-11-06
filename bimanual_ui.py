"""
BIMANUAL SPATIAL UI & WORD MANIPULATOR v6.0
Two-handed collaborative spatial computing interface for Smart Air Writer.

Roles:
- Dominant Hand (Right): Direct input, writing with pinch, spatial selection & dragging
- Non-Dominant Hand (Left): Spatial tool palette (Holographic floating radial menu)
- WordManipulator: Grab, translate, reposition, or fling-to-delete committed words using WordItem.offset_x/offset_y
"""

import math
import time
import numpy as np
import cv2
from gesture_engine import PointFilter2D


class RadialMenuItem:
    def __init__(self, item_id, label, icon_type, value, angle_deg, color=(255, 255, 255)):
        self.id = item_id
        self.label = label
        self.icon_type = icon_type  # "COLOR", "BRUSH", "ACTION"
        self.value = value
        self.angle_deg = angle_deg
        self.color = color
        self.radius_dist = 85.0
        self.hover_progress = 0.0


class RadialMenu:
    """
    Floating 3D Radial Palette displayed around the non-dominant palm.
    """
    def __init__(self):
        self.center_filter = PointFilter2D(mincutoff=1.5, beta=0.01)
        self.visible = False
        self.alpha_fade = 0.0
        self.items = [
            RadialMenuItem("col_1", "Cyan", "COLOR", "1", -90, (255, 240, 0)),
            RadialMenuItem("col_2", "Green", "COLOR", "2", -54, (0, 255, 128)),
            RadialMenuItem("col_3", "Yellow", "COLOR", "3", -18, (0, 230, 255)),
            RadialMenuItem("col_4", "Pink", "COLOR", "4", 18, (200, 50, 255)),
            RadialMenuItem("col_5", "White", "COLOR", "5", 54, (255, 255, 255)),
            RadialMenuItem("brush", "Brush", "BRUSH", "TOGGLE_BRUSH", 90, (0, 200, 255)),
            RadialMenuItem("board", "Board", "ACTION", "TOGGLE_BG", 126, (255, 180, 0)),
            RadialMenuItem("eraser", "Eraser", "ACTION", "TOGGLE_ERASER", 162, (0, 100, 255)),
            RadialMenuItem("undo", "Undo", "ACTION", "UNDO", 198, (255, 140, 0)),
            RadialMenuItem("clear", "Clear", "ACTION", "CLEAR", 234, (50, 50, 255)),
        ]
        self.last_trigger_time = 0.0

    def activate(self, palm_center, palm_radius=60.0):
        """Manual activation helper for testing or programmatic control."""
        self.visible = True
        self.alpha_fade = 1.0
        self._manual_center = palm_center

    def render(self, frame, cursor_pos=None):
        """Manual render helper."""
        cx, cy = getattr(self, '_manual_center', (200, 200))
        return self.update_and_render(frame, (cx, cy), True, cursor_pos or (0, 0))

    def check_selection(self, cursor_pos):
        """Checks if cursor intersects any menu button."""
        cx, cy = getattr(self, '_manual_center', (200, 200))
        for item in self.items:
            rad = math.radians(item.angle_deg)
            ix = int(cx + item.radius_dist * math.cos(rad))
            iy = int(cy + item.radius_dist * math.sin(rad))
            if math.hypot(cursor_pos[0] - ix, cursor_pos[1] - iy) < 26.0:
                return item.label
        return None

    def update_and_render(self, frame, palm_center, is_open_palm, cursor_pos, now=None):
        """
        Renders the holographic radial menu if open palm is held.
        Checks interaction with cursor_pos (dominant index or pinch).
        Returns: triggered_action (None or dict)
        """
        if now is None:
            now = time.time()

        if is_open_palm and palm_center is not None:
            cx, cy = self.center_filter.filter(palm_center[0], palm_center[1], now)
            self.alpha_fade = min(1.0, self.alpha_fade + 0.12)
        else:
            self.alpha_fade = max(0.0, self.alpha_fade - 0.15)
            self.center_filter.reset()
            return None

        if self.alpha_fade <= 0.05:
            return None

        triggered_action = None
        overlay = frame.copy()

        # Draw central floating ring around palm
        cv2.circle(overlay, (cx, cy), 42, (30, 30, 35), -1)
        cv2.circle(overlay, (cx, cy), 42, (0, 255, 180), 2)
        cv2.putText(overlay, "TOOLS", (cx - 24, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1)

        # Draw items in orbit
        for item in self.items:
            rad = math.radians(item.angle_deg)
            ix = int(cx + item.radius_dist * math.cos(rad))
            iy = int(cy + item.radius_dist * math.sin(rad))

            # Proximity to right-hand cursor
            dist_to_cursor = math.hypot(cursor_pos[0] - ix, cursor_pos[1] - iy) if cursor_pos else 999.0
            is_hovered = dist_to_cursor < 26.0

            if is_hovered:
                item.hover_progress = min(1.0, item.hover_progress + 0.20)
                if item.hover_progress >= 0.95 and (now - self.last_trigger_time > 0.6):
                    triggered_action = {"type": item.icon_type, "value": item.value, "label": item.label}
                    self.last_trigger_time = now
            else:
                item.hover_progress = max(0.0, item.hover_progress - 0.15)

            btn_r = int(18 + 5 * item.hover_progress)
            glow_col = item.color if item.icon_type == "COLOR" else (0, 240, 255)

            # Button background & border
            cv2.circle(overlay, (ix, iy), btn_r, (25, 25, 30), -1)
            cv2.circle(overlay, (ix, iy), btn_r, glow_col, 2 if not is_hovered else 3)

            # Content inside button
            if item.icon_type == "COLOR":
                cv2.circle(overlay, (ix, iy), btn_r - 5, item.color, -1)
            else:
                # Text initials
                txt = item.label[:3].upper()
                cv2.putText(overlay, txt, (ix - 12, iy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1)

            # Draw connector spoke
            cv2.line(overlay, (int(cx + 42 * math.cos(rad)), int(cy + 42 * math.sin(rad))),
                     (int(ix - btn_r * math.cos(rad)), int(iy - btn_r * math.sin(rad))),
                     (70, 70, 80), 1)

        # Alpha composite overlay
        cv2.addWeighted(overlay, self.alpha_fade, frame, 1.0 - self.alpha_fade, 0, frame)
        return triggered_action


class WordManipulator:
    """
    Allows spatial grabbing, moving, and fling-deleting committed words.
    Leverages WordItem.offset_x and WordItem.offset_y.
    """
    def __init__(self):
        self.selected_word = None
        self.grab_offset = (0, 0)
        self.initial_pinch = (0, 0)
        self.is_dragging = False

    def check_grab(self, words, pinch_pt, is_pinching):
        """
        If pinch started near a word's bounding box, grab it for translation.
        """
        px, py = pinch_pt

        if not is_pinching:
            # Release word
            if self.is_dragging and self.selected_word:
                # Check if flung off-screen or dragged to bottom trash area
                bbox = self.selected_word.get_bbox()
                if bbox and (bbox[1] + bbox[3] > 680 or bbox[0] < -50 or bbox[0] > 1300):
                    # Flag to delete
                    deleted_w = self.selected_word
                    self.selected_word = None
                    self.is_dragging = False
                    return ("DELETE", deleted_w)

            self.is_dragging = False
            self.selected_word = None
            return ("NONE", None)

        if self.is_dragging and self.selected_word:
            # Translate word offset
            self.selected_word.offset_x = px - self.grab_offset[0]
            self.selected_word.offset_y = py - self.grab_offset[1]
            return ("DRAGGING", self.selected_word)

        # Check if pinching over any existing word
        for w in reversed(words):
            if w.is_empty():
                continue
            bbox = w.get_bbox()
            if not bbox:
                continue
            bx, by, bw, bh = bbox
            # Bounding box with grab padding
            pad = 22
            if (bx - pad <= px <= bx + bw + pad) and (by - pad <= py <= by + bh + pad):
                self.selected_word = w
                self.is_dragging = True
                self.grab_offset = (px - w.offset_x, py - w.offset_y)
                return ("GRABBED", self.selected_word)

        return ("NONE", None)

    def draw_highlight(self, frame, word):
        """Draws a spatial bounding box around the grabbed word."""
        if not word or word.is_empty():
            return
        bbox = word.get_bbox()
        if not bbox:
            return
        bx, by, bw, bh = bbox
        pad = 8
        cv2.rectangle(frame, (bx - pad, by - pad), (bx + bw + pad, by + bh + pad), (0, 240, 255), 2)
        cv2.putText(frame, f"MOVE: {word.text if word.text else 'Word'}", (bx, by - pad - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1)
