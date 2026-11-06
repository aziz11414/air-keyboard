"""
SPLINE RENDERER PRO v6.0
High-fidelity geometric curve smoothing and vector rendering for Smart Air Writer.

Includes:
- Catmull-Rom to Cubic Bézier curve analytical conversion
- Multi-brush rendering (Neon, Calligraphy, Highlighter, Dotted, Standard)
- Professional SVG vector export with cubic Bézier `<path d="M... C...">`
"""

import math
import numpy as np
import cv2


def catmull_rom_to_bezier(p0, p1, p2, p3, tension=0.5):
    """
    Converts 4 Catmull-Rom control points to Cubic Bézier control points (B0, B1, B2, B3).
    Guarantees C1 continuity and passes exactly through P1 and P2.
    """
    x0, y0 = p0[0], p0[1]
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]

    # Tension factor
    alpha = (1.0 - tension) / 3.0

    # B0 = P1
    b0 = (x1, y1)
    # B1 = P1 + (P2 - P0) * alpha
    b1 = (x1 + (x2 - x0) * alpha, y1 + (y2 - y0) * alpha)
    # B2 = P2 - (P3 - P1) * alpha
    b2 = (x2 - (x3 - x1) * alpha, y2 - (y3 - y1) * alpha)
    # B3 = P2
    b3 = (x2, y2)

    return b0, b1, b2, b3


def sample_bezier_cubic(b0, b1, b2, b3, steps=10):
    """Samples points along a cubic Bézier curve for smooth raster rendering."""
    pts = []
    for i in range(steps + 1):
        t = i / float(steps)
        t2 = t * t
        t3 = t2 * t
        mt = 1.0 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        x = mt3 * b0[0] + 3 * mt2 * t * b1[0] + 3 * mt * t2 * b2[0] + t3 * b3[0]
        y = mt3 * b0[1] + 3 * mt2 * t * b1[1] + 3 * mt * t2 * b2[1] + t3 * b3[1]
        pts.append((int(round(x)), int(round(y))))
    return pts


def smooth_points_to_bezier_segments(points):
    """
    Takes a raw sequence of points and returns a list of cubic Bézier segments:
    [(B0, B1, B2, B3), ...]
    """
    if len(points) < 2:
        return []
    if len(points) == 2:
        # Linear as degenerate Bézier
        p0, p1 = points[0], points[1]
        return [((p0[0], p0[1]), (p0[0], p0[1]), (p1[0], p1[1]), (p1[0], p1[1]))]

    # Duplicate start and end to form Catmull-Rom boundary conditions
    extended = [points[0]] + list(points) + [points[-1]]
    segments = []
    for i in range(len(points) - 1):
        p0 = extended[i]
        p1 = extended[i + 1]
        p2 = extended[i + 2]
        p3 = extended[i + 3]
        seg = catmull_rom_to_bezier(p0, p1, p2, p3)
        segments.append(seg)
    return segments


def render_stroke_smooth(canvas, points, color, thickness, brush="NEON", brush_type=None):
    """
    Renders an entire stroke with Catmull-Rom smoothing and multi-brush aesthetics.
    """
    if brush_type is not None:
        brush = brush_type
    if len(points) < 2:
        if len(points) == 1:
            cv2.circle(canvas, (int(points[0][0]), int(points[0][1])), thickness // 2, color, -1)
        return

    segments = smooth_points_to_bezier_segments(points)

    if brush == "HIGHLIGHTER":
        overlay = canvas.copy()
        for b0, b1, b2, b3 in segments:
            sampled = sample_bezier_cubic(b0, b1, b2, b3, steps=8)
            for j in range(1, len(sampled)):
                cv2.line(overlay, sampled[j - 1], sampled[j], color, max(18, thickness * 2))
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
        return

    if brush == "CALLIGRAPHY":
        # Dynamic line width based on stroke tangent angle
        for b0, b1, b2, b3 in segments:
            sampled = sample_bezier_cubic(b0, b1, b2, b3, steps=10)
            for j in range(1, len(sampled)):
                p1, p2 = sampled[j - 1], sampled[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                angle = math.atan2(dy, dx)
                # Broader downstrokes, thinner horizontal strokes
                w_factor = abs(math.sin(angle - math.pi / 4.0))
                th = int(max(3, thickness * (0.4 + 1.2 * w_factor)))
                cv2.line(canvas, p1, p2, color, th, lineType=cv2.LINE_AA)
                cv2.circle(canvas, p2, th // 2, color, -1)
        return

    if brush == "DOTTED":
        dot_counter = 0
        for b0, b1, b2, b3 in segments:
            sampled = sample_bezier_cubic(b0, b1, b2, b3, steps=8)
            for pt in sampled:
                dot_counter += 1
                if dot_counter % 3 == 0:
                    cv2.circle(canvas, pt, max(2, thickness // 2), color, -1)
        return

    # Standard / NEON core rendering
    for b0, b1, b2, b3 in segments:
        sampled = sample_bezier_cubic(b0, b1, b2, b3, steps=8)
        for j in range(1, len(sampled)):
            cv2.line(canvas, sampled[j - 1], sampled[j], color, thickness, lineType=cv2.LINE_AA)
            cv2.circle(canvas, sampled[j], thickness // 2, color, -1)


def export_svg_bezier(strokes_list, width, height, filepath):
    """
    Exports strokes to a vector SVG file using true Cubic Bézier `<path d="M... C...">`.
    Produces infinitely scalable vector graphics compatible with Figma/Illustrator.
    """
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '  <defs>',
        '    <filter id="neon-glow" x="-20%" y="-20%" width="140%" height="140%">',
        '      <feGaussianBlur stdDeviation="4" result="blur" />',
        '      <feMerge>',
        '        <feMergeNode in="blur" />',
        '        <feMergeNode in="SourceGraphic" />',
        '      </feMerge>',
        '    </filter>',
        '  </defs>',
        f'  <rect width="{width}" height="{height}" fill="#121216" />'
    ]

    for stroke in strokes_list:
        if hasattr(stroke, 'points'):
            pts = stroke.points
            b, g, r = stroke.color
            th = stroke.thickness
            brush = getattr(stroke, 'brush', 'NEON')
        elif isinstance(stroke, (list, tuple)) and len(stroke) >= 1:
            pts = stroke[0]
            col = stroke[1] if len(stroke) > 1 else (255, 255, 255)
            b, g, r = col
            th = stroke[2] if len(stroke) > 2 else 3
            brush = stroke[3] if len(stroke) > 3 else 'NEON'
        elif isinstance(stroke, dict):
            pts = stroke.get('points', [])
            col = stroke.get('color', (255, 255, 255))
            b, g, r = col
            th = stroke.get('thickness', 3)
            brush = stroke.get('brush', 'NEON')
        else:
            continue

        if len(pts) < 2:
            continue

        hex_col = f"#{r:02x}{g:02x}{b:02x}"

        segments = smooth_points_to_bezier_segments(pts)
        if not segments:
            continue

        # Build path data
        first_p = segments[0][0]
        d_parts = [f"M {first_p[0]:.1f} {first_p[1]:.1f}"]

        for b0, b1, b2, b3 in segments:
            d_parts.append(f"C {b1[0]:.1f} {b1[1]:.1f}, {b2[0]:.1f} {b2[1]:.1f}, {b3[0]:.1f} {b3[1]:.1f}")

        path_d = " ".join(d_parts)

        if brush == "NEON":
            svg_lines.append(f'  <path d="{path_d}" stroke="{hex_col}" stroke-width="{th}" fill="none" stroke-linecap="round" stroke-linejoin="round" filter="url(#neon-glow)" />')
        elif brush == "HIGHLIGHTER":
            svg_lines.append(f'  <path d="{path_d}" stroke="{hex_col}" stroke-width="{th * 2}" fill="none" opacity="0.45" stroke-linecap="round" stroke-linejoin="round" />')
        else:
            svg_lines.append(f'  <path d="{path_d}" stroke="{hex_col}" stroke-width="{th}" fill="none" stroke-linecap="round" stroke-linejoin="round" />')

    svg_lines.append('</svg>')

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))
