# 🌐 Air Writer Web - React & Next.js / Browser Client

A production-ready web port of the AI Air Writer that runs 100% in the browser using client-side MediaPipe Hands.

---

## ⚡ Instant Run (Zero Setup)

You can launch the web client immediately with any local HTTP server (Python, VS Code Live Server, or npx):

### Option 1: Python Built-in Server
In this directory:
```bash
python -m http.server 3000
```
Open your browser at: **`http://localhost:3000`**

### Option 2: Node / npx serve
```bash
npx serve .
```

---

## 🚀 Features Included in the Web Client:
- **🖋️ Calligraphy Brush**: Dynamic velocity-based line thickness.
- **💡 Neon Glow**: Canvas shadow blur effect for cyberpunk neon light.
- **🖍️ Brushes**: Neon, Calligraphy, Rainbow (HSV cycle), Highlighter.
- **📄 Backgrounds**: Camera view, Clean Whiteboard, Dark Grid Slate.
- **🖐️ Gestures**:
  - ☝️ **Index Up**: 1.0s timer to engage writing.
  - ✌️ **2 Fingers**: Pen Up (Move cursor freely).
  - 🖐️ **Open Palm**: Sponge Eraser.
  - 👍 **Thumbs Up**: Instant Undo.
- **📸 High-Res PNG Snapshot**: One-click download of the canvas with or without video.

---

## ⚛️ Using in React / Next.js

Copy `AirWriterReact.jsx` into your Next.js `components/` directory:

```jsx
// app/page.jsx (or pages/index.jsx)
import dynamic from 'next/dynamic';

const AirWriter = dynamic(() => import('@/components/AirWriterReact'), {
  ssr: false, // Must run client-side for webcam & canvas
});

export default function Home() {
  return <AirWriter />;
}
```
Deploy directly to **Vercel**, **Netlify**, or **GitHub Pages**!
