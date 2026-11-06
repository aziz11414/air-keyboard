"""
VIRTUAL CAMERA BROADCASTER v6.0
Streams Smart Air Writer output to Zoom, Microsoft Teams, Google Meet, and Discord.

Uses pyvirtualcam to output frames to OBS Virtual Camera or Unity Capture.
Runs safely with graceful fallback if no virtual camera device is installed.
"""

import cv2
import numpy as np

try:
    import pyvirtualcam
    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False


class VirtualCameraManager:
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None
        self.is_active = False
        self.error_msg = ""
        self.device_name = ""

    def is_supported(self):
        return PYVIRTUALCAM_AVAILABLE

    def start(self):
        """Attempts to open virtual camera."""
        if not PYVIRTUALCAM_AVAILABLE:
            self.error_msg = "pyvirtualcam not installed (pip install pyvirtualcam)"
            return False, self.error_msg

        if self.is_active:
            return True, f"Virtual camera already streaming to {self.device_name}"

        try:
            # Try opening virtual camera (PixelFormat.BGR is standard for OpenCV)
            self.cam = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.BGR
            )
            self.is_active = True
            self.device_name = getattr(self.cam, "device", "Virtual Camera")
            self.error_msg = ""
            print(f"[V-CAM] Streaming active on device: {self.device_name}")
            return True, f"Streaming to {self.device_name}"
        except Exception as e:
            self.is_active = False
            self.cam = None
            err_str = str(e)
            if "OBS Virtual Camera device not found" in err_str:
                self.error_msg = "OBS Virtual Camera not found. Install OBS Studio or enable OBS Virtual Cam."
            else:
                self.error_msg = f"Virtual Cam error: {err_str[:60]}"
            print(f"[V-CAM] Failed to start: {self.error_msg}")
            return False, self.error_msg

    def stop(self):
        """Stops and closes virtual camera."""
        if self.cam:
            try:
                self.cam.close()
            except Exception:
                pass
            self.cam = None
        self.is_active = False
        return True, "Virtual camera stopped"

    def toggle(self):
        """Toggles between ON and OFF."""
        if self.is_active:
            return self.stop()
        else:
            return self.start()

    def send_frame(self, frame_bgr):
        """Sends a frame to virtual camera."""
        if not self.is_active or self.cam is None:
            return False

        try:
            # Ensure correct resolution
            h, w = frame_bgr.shape[:2]
            if w != self.width or h != self.height:
                frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

            self.cam.send(frame_bgr)
            return True
        except Exception as e:
            print(f"[V-CAM] Frame drop: {e}")
            return False
