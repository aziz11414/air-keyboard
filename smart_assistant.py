"""
SMART SESSION ASSISTANT & MARKDOWN EXPORTER v6.0
Automatic note synthesis, session transcription, and Markdown/PDF export for Smart Air Writer.

Includes:
- Structured Markdown note export with timestamps, statistics, and word lists
- Writing metrics: WPM (words per minute), total strokes, characters
- Multilingual glossary & offline translation matrix (French, English, Arabic)
"""

import os
import time
from datetime import datetime
from multilingual_engine import MultilingualManager


class SessionAssistant:
    def __init__(self, export_dir=None):
        self.export_dir = export_dir or os.path.join(os.getcwd(), "exports")
        os.makedirs(self.export_dir, exist_ok=True)
        self.committed_words = []  # [(word, timestamp)]
        self.drawn_shapes = []     # [(shape_type, timestamp)]
        self.session_start = time.time()

    def add_word(self, word):
        if word and word.strip():
            self.committed_words.append((word.strip(), time.time()))

    def add_shape(self, shape_name):
        if shape_name and shape_name != "NONE":
            self.drawn_shapes.append((shape_name, time.time()))

    def get_full_text(self):
        return " ".join(w[0] for w in self.committed_words)

    def calculate_stats(self):
        total_time_min = max(0.05, (time.time() - self.session_start) / 60.0)
        word_count = len(self.committed_words)
        char_count = sum(len(w[0]) for w in self.committed_words)
        wpm = round(word_count / total_time_min, 1)
        return {
            "words": word_count,
            "characters": char_count,
            "shapes": len(self.drawn_shapes),
            "duration_sec": int(time.time() - self.session_start),
            "wpm": wpm
        }

    def export_markdown(self):
        """
        Exports the entire session as a clean GitHub-flavored Markdown document.
        Returns: filepath
        """
        now_dt = datetime.now()
        filename = f"notes_{now_dt.strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(self.export_dir, filename)

        stats = self.calculate_stats()
        full_text = self.get_full_text()

        lines = [
            "# 📝 Smart Air Writer — Session Notes & AI Report",
            "",
            f"> **Date**: {now_dt.strftime('%A, %d %B %Y at %H:%M:%S')}",
            f"> **Duration**: {stats['duration_sec'] // 60}m {stats['duration_sec'] % 60}s | **Speed**: {stats['wpm']} WPM | **Words**: {stats['words']} | **Shapes**: {stats['shapes']}",
            "",
            "---",
            "",
            "## 📌 Full Text Transcript",
            "",
            f"**{full_text}**" if full_text else "*(No text committed during this session)*",
            "",
            "---",
            "",
            "## ⏱️ Chronological Timeline",
            "",
            "| Time | Recognized Word / Event |",
            "|---|---|"
        ]

        if self.committed_words:
            t0 = self.session_start
            for word, t in self.committed_words:
                rel = int(t - t0)
                m, s = rel // 60, rel % 60
                lines.append(f"| `{m:02d}:{s:02d}` | **{word}** |")
        else:
            lines.append("| `00:00` | *(Empty session)* |")

        if self.drawn_shapes:
            lines.extend([
                "",
                "## 📐 Detected Shapes & Diagrams",
                "",
                "| Time | Shape Type | Status |",
                "|---|---|---|"
            ])
            for shape, t in self.drawn_shapes:
                rel = int(t - self.session_start)
                m, s = rel // 60, rel % 60
                lines.append(f"| `{m:02d}:{s:02d}` | `{shape}` | Geometric Snap OK |")

        # Multilingual vocabulary matrix for recognized words
        if self.committed_words:
            unique_words = list(dict.fromkeys(w[0].upper() for w in self.committed_words))
            glossary_rows = []
            for w in unique_words:
                entry = MultilingualManager.get_glossary_entry(w)
                if entry:
                    glossary_rows.append(f"| **{w}** | {entry.get('FR', '-')} | {entry.get('EN', '-')} | {entry.get('AR', '-')} |")
                else:
                    glossary_rows.append(f"| **{w}** | *{w}* | *{w}* | *{w}* |")

            lines.extend([
                "",
                "## 🌐 Multilingual Analysis (FR / EN / AR)",
                "",
                "| Word | Français (FR) | English (EN) | العربية (AR) |",
                "|---|---|---|---|"
            ])
            lines.extend(glossary_rows)

        lines.extend([
            "",
            "---",
            "*Generated automatically by Smart Air Writer v6.0 PRO — Open Source Computer Vision.*"
        ])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"[ASSISTANT] Session notes saved to: {filepath}")
        return filepath
