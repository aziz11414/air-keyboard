"""
SMART SHAPE ENGINE & GEOMETRIC SNAP v6.0
Automatic geometric shape recognition and snapping for Smart Air Writer.

Recognizes hand-drawn strokes into perfect geometric primitives:
- LINE: Straight line connecting start and end
- CIRCLE / ELLIPSE: Perfect circular or elliptical contour
- RECTANGLE / SQUARE: Closed 4-sided polygon with right angles
- TRIANGLE: Closed 3-sided polygon
- ARROW: Directed straight arrow with head
"""

import math
import numpy as np
import cv2


class ShapeRecognizer:
    """
    Analyzes raw stroke points and detects if they match a geometric primitive.
    """

    @staticmethod
    def recognize_and_snap(points, min_points=8):
        """
        Takes raw points [(x, y, ...), ...] and returns:
        (shape_type, snapped_points, confidence)
        If no shape recognized, returns ("NONE", points, 0.0)
        """
        if len(points) < min_points:
            return "NONE", points, 0.0

        # Extract 2D coordinates
        coords = np.array([(p[0], p[1]) for p in points], dtype=np.float32)
        n = len(coords)

        # 1. Total arc length & start-to-end distance
        dists = np.hypot(coords[1:, 0] - coords[:-1, 0], coords[1:, 1] - coords[:-1, 1])
        total_len = float(np.sum(dists))
        if total_len < 20.0:
            return "NONE", points, 0.0

        start_pt = coords[0]
        end_pt = coords[-1]
        chord_len = float(math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1]))
        is_closed = (chord_len / total_len) < 0.22

        # ---------------------------------------------------------------------
        # A. STRAIGHT LINE DETECTION
        # ---------------------------------------------------------------------
        if not is_closed and (chord_len / total_len) > 0.90:
            # Measure max perpendicular deviation from start-to-end line
            v = end_pt - start_pt
            v_len = math.hypot(v[0], v[1]) + 1e-5
            v_norm = v / v_len

            diffs = coords - start_pt
            # Cross product gives perpendicular distance
            perp_dists = np.abs(diffs[:, 0] * v_norm[1] - diffs[:, 1] * v_norm[0])
            max_dev = float(np.max(perp_dists))

            if max_dev < 18.0 or (max_dev / total_len) < 0.08:
                # Snap to straight line (resampled with original step count)
                steps = max(5, min(20, n))
                snapped = []
                for t in np.linspace(0.0, 1.0, steps):
                    pt = start_pt * (1.0 - t) + end_pt * t
                    snapped.append((float(pt[0]), float(pt[1])))
                return "LINE", snapped, 0.95

        # ---------------------------------------------------------------------
        # B. ARROW DETECTION (Line with a sharp hook/head at the end)
        # ---------------------------------------------------------------------
        if not is_closed and n >= 12 and total_len > 60.0:
            # Check if 75% of stroke is a straight shaft
            shaft_idx = int(n * 0.75)
            shaft_start = coords[0]
            shaft_end = coords[shaft_idx]
            shaft_chord = math.hypot(shaft_end[0] - shaft_start[0], shaft_end[1] - shaft_start[1])
            shaft_len = float(np.sum(dists[:shaft_idx]))

            if shaft_len > 0 and (shaft_chord / shaft_len) > 0.88:
                # Check if tail bends sharply
                tail_v = coords[-1] - shaft_end
                shaft_v = shaft_end - shaft_start
                shaft_mag = math.hypot(shaft_v[0], shaft_v[1]) + 1e-5
                tail_mag = math.hypot(tail_v[0], tail_v[1]) + 1e-5

                dot = (shaft_v[0] * tail_v[0] + shaft_v[1] * tail_v[1]) / (shaft_mag * tail_mag)
                if dot < 0.5:  # sharp turn backward like an arrowhead
                    # Snap: draw clean straight shaft + 2 arrow wings
                    shaft_pts = [
                        (float(shaft_start[0]), float(shaft_start[1])),
                        (float(shaft_end[0]), float(shaft_end[1]))
                    ]
                    # Arrowhead wings
                    angle = math.atan2(shaft_v[1], shaft_v[0])
                    head_len = min(28.0, shaft_mag * 0.28)
                    wing1 = (
                        float(shaft_end[0] - head_len * math.cos(angle - 0.55)),
                        float(shaft_end[1] - head_len * math.sin(angle - 0.55))
                    )
                    wing2 = (
                        float(shaft_end[0] - head_len * math.cos(angle + 0.55)),
                        float(shaft_end[1] - head_len * math.sin(angle + 0.55))
                    )
                    # Continuous polyline representing arrow
                    arrow_snapped = [
                        shaft_pts[0], shaft_pts[1],
                        wing1, shaft_pts[1], wing2
                    ]
                    return "ARROW", arrow_snapped, 0.90

        # ---------------------------------------------------------------------
        # C. CLOSED SHAPES: CIRCLE, RECTANGLE, TRIANGLE
        # ---------------------------------------------------------------------
        if is_closed:
            pts_i32 = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
            peri = cv2.arcLength(pts_i32, True)
            area = cv2.contourArea(pts_i32)

            if peri > 40.0 and area > 100.0:
                # 1. Polygon approximation with Douglas-Peucker
                eps = 0.040 * peri
                approx = cv2.approxPolyDP(pts_i32, eps, True)
                num_vertices = len(approx)

                # Radius variance from centroid
                cx = float(np.mean(coords[:, 0]))
                cy = float(np.mean(coords[:, 1]))
                radii = np.hypot(coords[:, 0] - cx, coords[:, 1] - cy)
                mean_r = float(np.mean(radii))
                std_r = float(np.std(radii))
                r_cov = (std_r / mean_r) if mean_r > 0 else 1.0

                # TRIANGLE: 3 vertices
                if num_vertices == 3:
                    tri_pts = [tuple(map(float, pt[0])) for pt in approx]
                    tri_pts.append(tri_pts[0])  # close triangle
                    return "TRIANGLE", tri_pts, 0.90

                # RECTANGLE / QUAD: 4 vertices
                if num_vertices == 4:
                    rect_pts = [tuple(map(float, pt[0])) for pt in approx]
                    rect_pts.append(rect_pts[0])  # close rectangle
                    return "RECTANGLE", rect_pts, 0.92

                # CIRCLE: radius variance is small (< 12% of radius) and vertices >= 5
                if r_cov < 0.14:
                    (mcx, mcy), radius = cv2.minEnclosingCircle(pts_i32)
                    steps = 36
                    snapped = []
                    for i in range(steps + 1):
                        theta = (2.0 * math.pi * i) / steps
                        px = mcx + radius * math.cos(theta)
                        py = mcy + radius * math.sin(theta)
                        snapped.append((float(px), float(py)))
                    return "CIRCLE", snapped, 0.95

                # ELLIPSE: moderate radius variance and len >= 5
                if 0.14 <= r_cov < 0.28 and len(pts_i32) >= 5:
                    ellipse = cv2.fitEllipse(pts_i32)
                    (ecx, ecy), (d1, d2), angle = ellipse
                    rx, ry = d1 / 2.0, d2 / 2.0
                    ang_rad = math.radians(angle)
                    cos_a, sin_a = math.cos(ang_rad), math.sin(ang_rad)

                    steps = 36
                    snapped = []
                    for i in range(steps + 1):
                        t = (2.0 * math.pi * i) / steps
                        ex = rx * math.cos(t)
                        ey = ry * math.sin(t)
                        rx_rot = ex * cos_a - ey * sin_a + ecx
                        ry_rot = ex * sin_a + ey * cos_a + ecy
                        snapped.append((float(rx_rot), float(ry_rot)))
                    return "ELLIPSE", snapped, 0.88

        return "NONE", points, 0.0
