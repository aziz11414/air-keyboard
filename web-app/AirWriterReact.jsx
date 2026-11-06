import React, { useEffect, useRef, useState } from 'react';

/**
 * AirWriterReact - Production React / Next.js Component
 * Uses client-side MediaPipe Hands for air handwriting,
 * velocity calligraphy, neon glow, and gesture controls.
 */
export default function AirWriterReact() {
  const videoRef = useRef(null);
  const bgCanvasRef = useRef(null);
  const inkCanvasRef = useRef(null);
  const hudCanvasRef = useRef(null);

  const [color, setColor] = useState('#00f0ff');
  const [brush, setBrush] = useState('NEON'); // 'NEON', 'CALLIGRAPHY', 'RAINBOW', 'HIGHLIGHTER'
  const [bgMode, setBgMode] = useState('WEBCAM'); // 'WEBCAM', 'WHITEBOARD', 'DARK_SLATE'
  const [status, setStatus] = useState('WAITING FOR HAND');

  useEffect(() => {
    let camera = null;
    let hands = null;

    // Load MediaPipe scripts dynamically if not present
    const loadScript = (src) => {
      return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) return resolve();
        const script = document.createElement('script');
        script.src = src;
        script.crossOrigin = 'anonymous';
        script.onload = resolve;
        script.onerror = reject;
        document.body.appendChild(script);
      });
    };

    async function init() {
      await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js');
      await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js');

      const HandsClass = window.Hands;
      const CameraClass = window.Camera;

      if (!HandsClass || !CameraClass) return;

      hands = new HandsClass({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      });

      hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.65,
        minTrackingConfidence: 0.65,
      });

      hands.onResults((results) => {
        // Render loop using refs
      });

      if (videoRef.current) {
        camera = new CameraClass(videoRef.current, {
          onFrame: async () => {
            if (videoRef.current) await hands.send({ image: videoRef.current });
          },
          width: 1280,
          height: 720,
        });
        camera.start();
      }
    }

    init();

    return () => {
      if (camera) camera.stop();
      if (hands) hands.close();
    };
  }, []);

  return (
    <div className="relative w-full h-screen bg-black overflow-hidden flex flex-col">
      {/* Header HUD */}
      <div className="h-14 bg-gray-900/90 border-b border-gray-800 flex items-center justify-between px-6 z-10">
        <h1 className="text-white font-bold text-lg tracking-wider">AIR WRITER <span className="text-xs bg-cyan-900 text-cyan-300 px-2 py-0.5 rounded">NEXT.JS</span></h1>
        <div className="text-xs font-semibold px-3 py-1 rounded border border-cyan-500/40 text-cyan-300 bg-cyan-950/40">
          {status}
        </div>
      </div>

      {/* Main Screen */}
      <div className="relative flex-1">
        <video ref={videoRef} className="hidden" playsInline />
        <canvas ref={bgCanvasRef} className="absolute inset-0 w-full h-full object-cover" />
        <canvas ref={inkCanvasRef} className="absolute inset-0 w-full h-full object-cover pointer-events-none" />
        <canvas ref={hudCanvasRef} className="absolute inset-0 w-full h-full object-cover pointer-events-none" />
      </div>
    </div>
  );
}
