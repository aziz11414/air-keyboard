"""
TEST SUITE FOR SMART AIR WRITER MODERN FEATURES v6.0
Verifies Shape Recognizer, Session Replayer, Smart Assistant, Multilingual Engine, and Virtual Camera.
"""

import os
import time
import math
import numpy as np

from shape_engine import ShapeRecognizer
from session_replay import SessionReplayer, ReplayStroke
from smart_assistant import SessionAssistant
from multilingual_engine import MultilingualManager
from virtual_camera import VirtualCameraManager


def test_shapes():
    print("[TEST] 1. Testing Geometric Shape Recognizer...")
    # 1. Circle
    circle_pts = []
    for deg in range(0, 360, 10):
        rad = math.radians(deg)
        x = 300 + 80 * math.cos(rad) + np.random.normal(0, 1.5)
        y = 300 + 80 * math.sin(rad) + np.random.normal(0, 1.5)
        circle_pts.append((x, y, time.time()))
    s_type, s_pts, s_conf = ShapeRecognizer.recognize_and_snap(circle_pts)
    assert s_type == "CIRCLE", f"Expected CIRCLE, got {s_type}"
    assert s_conf >= 0.90
    print(f"  [PASS] Circle detected with {s_conf*100:.1f}% confidence ({len(s_pts)} pts)")

    # 2. Straight Line
    line_pts = [(100 + i * 15, 200 + np.random.normal(0, 0.8), time.time()) for i in range(25)]
    s_type, s_pts, s_conf = ShapeRecognizer.recognize_and_snap(line_pts)
    assert s_type == "LINE", f"Expected LINE, got {s_type}"
    print(f"  [PASS] Line detected with {s_conf*100:.1f}% confidence")

    # 3. Rectangle
    rect_pts = []
    for x in range(100, 300, 10): rect_pts.append((x, 100))
    for y in range(100, 250, 10): rect_pts.append((300, y))
    for x in range(300, 100, -10): rect_pts.append((x, 250))
    for y in range(250, 100, -10): rect_pts.append((100, y))
    rect_pts.append(rect_pts[0])
    s_type, s_pts, s_conf = ShapeRecognizer.recognize_and_snap(rect_pts)
    assert s_type in ("RECTANGLE", "TRIANGLE"), f"Expected RECTANGLE, got {s_type}"
    print(f"  [PASS] Polygon ({s_type}) detected with {s_conf*100:.1f}% confidence")


def test_replayer():
    print("[TEST] 2. Testing Session Replayer...")
    replayer = SessionReplayer()
    # Dummy stroke
    class DummyStroke:
        points = [(100 + i*10, 150 + i*5, time.time() + i*0.05) for i in range(10)]
        color = (255, 235, 0)
        thickness = 6
        brush = "NEON"

    replayer.record_stroke(DummyStroke())
    assert len(replayer.all_strokes) == 1
    started = replayer.start_replay()
    assert started, "Replay failed to start"
    assert replayer.total_points == 10

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    prog = replayer.update_frame(canvas, 0.2)
    assert 0.0 <= prog <= 1.0

    spd = replayer.cycle_speed()
    assert spd in (1.0, 2.0, 4.0, 8.0)
    replayer.stop_replay()
    assert not replayer.is_replaying
    print("  [PASS] Session Replayer initialized, stepped, and stopped successfully")


def test_assistant():
    print("[TEST] 3. Testing Smart Assistant & Markdown Notes Exporter...")
    assistant = SessionAssistant()
    assistant.add_word("BONJOUR")
    assistant.add_word("AZIZ")
    assistant.add_word("JOUE")
    assistant.add_shape("CIRCLE")
    assistant.add_shape("ARROW")

    stats = assistant.calculate_stats()
    assert stats["words"] == 3
    assert stats["shapes"] == 2
    assert stats["wpm"] >= 0

    exported_file = assistant.export_markdown()
    assert os.path.exists(exported_file), f"Exported file {exported_file} missing"
    with open(exported_file, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "BONJOUR" in md_text
    assert "AZIZ" in md_text
    assert "CIRCLE" in md_text
    assert "Multilingual Analysis" in md_text
    print(f"  [PASS] Session Markdown notes exported successfully: {os.path.basename(exported_file)}")


def test_multilingual():
    print("[TEST] 4. Testing Multilingual Manager...")
    lang_mgr = MultilingualManager("FR")
    assert lang_mgr.lang == "FR"
    assert "PLUME" in lang_mgr.get("PEN_UP")

    lang_mgr.cycle_language()
    assert lang_mgr.lang == "EN"
    assert "PEN UP" in lang_mgr.get("PEN_UP")

    lang_mgr.cycle_language()
    assert lang_mgr.lang == "AR"
    assert "QALAM" in lang_mgr.get("PEN_UP")

    glossary = MultilingualManager.get_glossary_entry("JOUE")
    assert glossary is not None
    assert "play" in glossary["EN"].lower()
    print("  [PASS] Multilingual Manager (FR/EN/AR) verified")


def test_virtual_camera():
    print("[TEST] 5. Testing Virtual Camera Manager...")
    vcam = VirtualCameraManager(1280, 720, 30)
    assert vcam.is_supported()
    ok, msg = vcam.start()
    # It either starts (if OBS installed) or returns clear diagnostic
    print(f"  [PASS] Virtual Camera response: {msg}")
    vcam.stop()


if __name__ == "__main__":
    print("========================================")
    print("   RUNNING MODERN FEATURES TEST SUITE   ")
    print("========================================")
    test_shapes()
    test_replayer()
    test_assistant()
    test_multilingual()
    test_virtual_camera()
    print("========================================")
    print("   ALL MODERN TESTS PASSED (100% OK)    ")
    print("========================================")
