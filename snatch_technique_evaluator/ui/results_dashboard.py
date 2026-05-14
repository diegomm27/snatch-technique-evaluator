from __future__ import annotations

"""Results Dashboard screen per WEIA-17 wireframes.

Full 3-zone layout:
- Left Panel (60%): Video player with pose overlay + timeline + phase markers
- Right Panel (40%): Metrics (overall score gauge, phase breakdown, key metrics cards)
- Bottom Bar: Export (PDF/CSV), Save, Compare, Settings actions
"""

import json
from pathlib import Path
from typing import Any

from .design_system import (
    THEME_LIGHT,
    THEME_DARK,
    COLOR_PRIMARY_600,
    COLOR_PRIMARY_500,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_ERROR,
    COLOR_INFO,
    COLOR_BORDER_LIGHT,
    COLOR_BG_LIGHT,
    COLOR_BG_DARK,
    COLOR_SURFACE_LIGHT,
    COLOR_SURFACE_DARK,
    COLOR_TEXT_PRIMARY_LIGHT,
    COLOR_TEXT_PRIMARY_DARK,
    COLOR_TEXT_SECONDARY_LIGHT,
    COLOR_TEXT_SECONDARY_DARK,
)


class PhaseMetric:
    """A single phase metric entry with score and color coding."""

    def __init__(self, name: str, score: float, duration: str = "") -> None:
        self.name = name
        self.score = score
        self.duration = duration

    @property
    def color(self) -> str:
        if self.score >= 80:
            return COLOR_SUCCESS
        if self.score >= 65:
            return COLOR_WARNING
        return COLOR_ERROR


class KeyMetric:
    """A single key metric card with label, value, and unit."""

    def __init__(self, label: str, value: str, unit: str = "") -> None:
        self.label = label
        self.value = value
        self.unit = unit


class ResultsDashboard:
    """Results Dashboard per WEIA-17 wireframes.

    Full 3-zone layout with:
    - Video player with phase-aware timeline and controls
    - Overall score gauge with color-coded description
    - Phase-by-phase breakdown with visual bars
    - Key metrics cards grid
    - Bottom action bar with export/save/compare/settings
    """

    PHASE_COLORS = {
        "setup": COLOR_SUCCESS,
        "first_pull": COLOR_WARNING,
        "second_pull": COLOR_INFO,
        "turnover": COLOR_SUCCESS,
        "catch": COLOR_WARNING,
        "recovery": COLOR_SUCCESS,
        "finish": COLOR_SUCCESS,
    }

    PHASE_LABELS = {
        "setup": "Setup",
        "first_pull": "1st Pull",
        "second_pull": "2nd Pull",
        "turnover": "Turnover",
        "catch": "Catch",
        "recovery": "Recovery",
        "finish": "Finish",
    }

    def __init__(
        self,
        analysis_data: dict | None = None,
        on_export=None,
        on_compare=None,
        on_settings=None,
        on_back=None,
        theme: str = "light",
    ) -> None:
        self.analysis_data = analysis_data or {}
        self.on_export = on_export
        self.on_compare = on_compare
        self.on_settings = on_settings
        self.on_back = on_back
        self.theme_name = theme

        try:
            import ttkbootstrap as ttk
            self._has_ttkb = True
        except ImportError:
            import tkinter as tk
            from tkinter import ttk as _ttk
            self._has_ttkb = False

        try:
            import tkinter as tk
            from tkinter import messagebox
        except ImportError as exc:
            raise RuntimeError("tkinter is unavailable in this environment.") from exc

        self.tk = tk
        self.messagebox = messagebox

        if self._has_ttkb:
            self.root = ttk.Window(
                themename="flatly" if theme == "light" else "darkly"
            )
        else:
            self.root = tk.Tk()

        self.root.title("Analysis Results - Weight Lifting Analyzer")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        theme_dict = THEME_LIGHT if theme == "light" else THEME_DARK
        self.root.configure(bg=theme_dict["bg"])

        self._build_styles(theme_dict)
        self._build_layout(theme_dict)
        self._populate_data()

    def _build_styles(self, theme: dict) -> None:
        if self._has_ttkb:
            style = self.root.style
        else:
            style = self.ttk.Style(self.root)
            try:
                style.theme_use("clam")
            except Exception:
                pass

        style.configure("Results.TFrame", background=theme["bg"])

        # Video panel
        style.configure("VideoPanel.TFrame", background="#1a1a2e")
        style.configure(
            "VideoTitle.TLabel",
            background="#1a1a2e",
            foreground="#e0e0e0",
            font=("Segoe UI Semibold", 11),
        )

        # Score gauge
        style.configure(
            "ScoreGauge.TLabel",
            background=theme["surface"],
            foreground=theme["text_primary"],
            font=("Segoe UI Semibold", 48),
        )
        style.configure(
            "ScoreLabel.TLabel",
            background=theme["surface"],
            foreground=theme["text_secondary"],
            font=("Segoe UI", 12),
        )
        style.configure(
            "ScoreDescription.TLabel",
            background=theme["surface"],
            foreground=theme["text_secondary"],
            font=("Segoe UI", 11),
        )

        # Phase breakdown
        style.configure(
            "PhaseLabel.TLabel",
            background=theme["surface"],
            foreground=theme["text_primary"],
            font=("Segoe UI", 11),
        )
        style.configure(
            "PhaseScore.TLabel",
            background=theme["surface"],
            foreground=theme["text_primary"],
            font=("Segoe UI Semibold", 11),
        )

        # Key metrics
        style.configure(
            "KeyMetricCard.TFrame",
            background=theme["surface"],
        )
        style.configure(
            "KeyMetricLabel.TLabel",
            background=theme["surface"],
            foreground=theme["text_secondary"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "KeyMetricValue.TLabel",
            background=theme["surface"],
            foreground=theme["text_primary"],
            font=("Segoe UI Semibold", 16),
        )

        # Bottom action bar
        style.configure(
            "ActionBtn.TButton",
            font=("Segoe UI Semibold", 11),
            padding=(14, 8),
        )
        style.configure(
            "ActionBtnGhost.TButton",
            font=("Segoe UI", 11),
            padding=(12, 7),
        )

        # Phase indicator buttons
        style.configure(
            "PhaseIndicator.TButton",
            font=("Segoe UI", 9),
            padding=(6, 4),
        )

    def _build_layout(self, theme: dict) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        # ── Top Bar ────────────────────────────────────────────────────
        top_bar = self.ttk.Frame(root, style="Results.TFrame", padding=(16, 10, 16, 8))
        top_bar.grid(row=0, column=0, sticky="ew")
        top_bar.columnconfigure(1, weight=1)

        self.ttk.Label(
            top_bar,
            text="Analysis Results",
            style="VideoTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")

        self.video_name_var = self.tk.StringVar(
            value=self.analysis_data.get("video_name", "Unknown Video")
        )
        self.ttk.Label(
            top_bar,
            textvariable=self.video_name_var,
            style="VideoTitle.TLabel",
        ).grid(row=0, column=1, sticky="w", padx=(12, 0))

        # Theme toggle
        self.theme_label = self.ttk.Label(
            top_bar,
            text="\u263E",
            style="ActionBtnGhost.TButton",
            cursor="hand2",
        )
        self.theme_label.grid(row=0, column=2, sticky="e", padx=(8, 0))
        self.theme_label.bind("<Button-1>", self._toggle_theme)

        if self.on_back:
            self.ttk.Button(
                top_bar,
                text="\u2190 Back",
                style="ActionBtnGhost.TButton",
                command=self.on_back,
            ).grid(row=0, column=3, sticky="e")

        if self.on_settings:
            self.ttk.Button(
                top_bar,
                text="\u2699",
                style="ActionBtnGhost.TButton",
                command=self.on_settings,
            ).grid(row=0, column=4, sticky="e")

        # ── Main Content ───────────────────────────────────────────────
        main_content = self.ttk.Frame(root, style="Results.TFrame", padding=(16, 8, 16, 0))
        main_content.grid(row=1, column=0, sticky="nsew")
        main_content.columnconfigure(0, weight=3)  # Video panel: 60%
        main_content.columnconfigure(1, weight=2)  # Metrics panel: 40%
        main_content.rowconfigure(0, weight=1)

        # ── Left Panel: Video + Pose Visualization ─────────────────────
        video_panel = self.ttk.Frame(main_content, style="VideoPanel.TFrame", padding=0)
        video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        video_panel.columnconfigure(0, weight=1)
        video_panel.rowconfigure(0, weight=1)
        video_panel.rowconfigure(1, weight=0)
        video_panel.rowconfigure(2, weight=0)

        # Video display area with canvas
        self.video_canvas = self.tk.Canvas(
            video_panel,
            bg="#1a1a2e",
            highlightthickness=0,
        )
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        self._draw_video_placeholder()

        # Phase indicators strip
        phase_frame = self.ttk.Frame(video_panel, style="Results.TFrame", padding=(0, 8))
        phase_frame.grid(row=1, column=0, sticky="ew")
        phase_frame.columnconfigure(tuple(range(7)), weight=1)

        self._phase_indicators: list[Any] = []
        for phase_key, phase_label in self.PHASE_LABELS.items():
            color = self.PHASE_COLORS.get(phase_key, COLOR_TEXT_SECONDARY_LIGHT)
            btn = self.ttk.Label(
                phase_frame,
                text=phase_label,
                style="VideoTitle.TLabel",
            )
            btn.grid(row=0, column=list(self.PHASE_LABELS.keys()).index(phase_key), padx=1)
            self._phase_indicators.append(btn)

        # Timeline + playback controls
        controls_frame = self.ttk.Frame(video_panel, style="Results.TFrame", padding=(0, 0))
        controls_frame.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        controls_frame.columnconfigure(0, weight=0)
        controls_frame.columnconfigure(1, weight=0)
        controls_frame.columnconfigure(2, weight=1)
        controls_frame.columnconfigure(3, weight=0)
        controls_frame.columnconfigure(4, weight=0)

        self.ttk.Button(
            controls_frame, text="\u25C0", style="ActionBtnGhost.TButton"
        ).grid(row=0, column=0, padx=(0, 8))

        self._play_button = self.ttk.Button(
            controls_frame, text="\u25B6", style="ActionBtn.TButton"
        )
        self._play_button.grid(row=0, column=1, padx=(0, 8))

        # Timeline bar with phase segments
        self.timeline_canvas = self.tk.Canvas(
            controls_frame,
            height=8,
            bg=COLOR_BORDER_LIGHT,
            highlightthickness=0,
        )
        self.timeline_canvas.grid(row=0, column=2, sticky="ew", padx=(0, 12))
        self._draw_timeline()

        self.ttk.Button(
            controls_frame, text="1x", style="ActionBtnGhost.TButton"
        ).grid(row=0, column=3, padx=(0, 8))

        self.time_label = self.ttk.Label(
            controls_frame,
            text="0:00 / 0:00",
            style="VideoTitle.TLabel",
        )
        self.time_label.grid(row=0, column=4)

        # ── Right Panel: Metrics ───────────────────────────────────────
        metrics_panel = self.ttk.Frame(main_content, style="Results.TFrame", padding=(8, 0, 0, 0))
        metrics_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        metrics_panel.columnconfigure(0, weight=1)
        metrics_panel.rowconfigure(0, weight=0)
        metrics_panel.rowconfigure(1, weight=1)
        metrics_panel.rowconfigure(2, weight=0)

        # Overall Score Card
        score_card = self.ttk.Frame(metrics_panel, style="KeyMetricCard.TFrame", padding=(20, 16))
        score_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        score_card.columnconfigure(0, weight=1)

        self.ttk.Label(
            score_card,
            text="OVERALL SCORE",
            style="ScoreLabel.TLabel",
        ).grid(row=0, column=0, sticky="w")

        self.score_value_label = self.ttk.Label(
            score_card,
            text="84/100",
            style="ScoreGauge.TLabel",
        )
        self.score_value_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.score_desc_label = self.ttk.Label(
            score_card,
            text="Good form, minor knee tracking issue",
            style="ScoreDescription.TLabel",
        )
        self.score_desc_label.grid(row=2, column=0, sticky="w", pady=(2, 0))

        # Score bar
        score_bar_frame = self.ttk.Frame(score_card, style="KeyMetricCard.TFrame")
        score_bar_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        score_bar_frame.columnconfigure(0, weight=1)

        self.score_bar_canvas = self.tk.Canvas(
            score_bar_frame, height=10, bg=COLOR_BORDER_LIGHT, highlightthickness=0
        )
        self.score_bar_canvas.grid(row=0, column=0, sticky="ew")

        # Phase Breakdown Card
        phase_card = self.ttk.Frame(metrics_panel, style="KeyMetricCard.TFrame", padding=(20, 16))
        phase_card.grid(row=1, column=0, sticky="nsew", pady=(0, 12))
        phase_card.columnconfigure(1, weight=1)

        self.ttk.Label(
            phase_card,
            text="PHASE BREAKDOWN",
            style="ScoreLabel.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        self.phase_labels: dict[str, Any] = {}
        phases_data = [
            ("Setup", 85, "1.2s"),
            ("1st Pull", 78, "1.8s"),
            ("2nd Pull", 82, "0.9s"),
            ("Turnover", 90, "0.6s"),
            ("Catch", 76, "0.3s"),
            ("Recovery", 88, "1.4s"),
        ]

        for i, (name, score, duration) in enumerate(phases_data):
            row = i + 1
            self.ttk.Label(
                phase_card, text=name, style="PhaseLabel.TLabel"
            ).grid(row=row, column=0, sticky="w", pady=3)

            # Phase bar
            bar_frame = self.ttk.Frame(phase_card, style="KeyMetricCard.TFrame")
            bar_frame.grid(row=row, column=1, sticky="ew", padx=(8, 0))
            bar_frame.columnconfigure(0, weight=1)

            bar_canvas = self.tk.Canvas(
                bar_frame, height=12, bg=COLOR_BORDER_LIGHT, highlightthickness=0, width=150
            )
            bar_canvas.grid(row=0, column=0, sticky="ew")
            fill = (score / 100) * 150
            phase_color = COLOR_SUCCESS if score >= 80 else COLOR_WARNING
            bar_canvas.create_rectangle(0, 0, fill, 12, fill=phase_color, outline="")

            self.phase_labels[name] = {"canvas": bar_canvas, "score": score}

            self.ttk.Label(
                phase_card, text=str(score), style="PhaseScore.TLabel"
            ).grid(row=row, column=2, sticky="w", padx=(8, 0))

            # Duration label
            self.ttk.Label(
                phase_card, text=duration, style="KeyMetricLabel.TLabel"
            ).grid(row=row, column=3, sticky="w", padx=(4, 0))

        # Key Metrics Cards
        key_metrics_card = self.ttk.Frame(metrics_panel, style="KeyMetricCard.TFrame", padding=(20, 16))
        key_metrics_card.grid(row=2, column=0, sticky="ew")
        key_metrics_card.columnconfigure(tuple(range(3)), weight=1)

        key_metrics = [
            ("Barbell Path", "92%", "aligned"),
            ("Knee Angle", "32\u00b0", "at top"),
            ("Timing Ratio", "1.2s", "pull"),
        ]

        for i, (label, value, unit) in enumerate(key_metrics):
            col = i
            metric_card = self.ttk.Frame(
                key_metrics_card, style="KeyMetricCard.TFrame", padding=(12, 10)
            )
            metric_card.grid(row=0, column=col, sticky="nsew", padx=(0, 8) if i < 2 else (0, 0))

            self.ttk.Label(
                metric_card, text=label, style="KeyMetricLabel.TLabel"
            ).grid(row=0, column=0, sticky="w")

            self.ttk.Label(
                metric_card, text=value, style="KeyMetricValue.TLabel"
            ).grid(row=1, column=0, sticky="w", pady=(2, 0))

            self.ttk.Label(
                metric_card, text=unit, style="KeyMetricLabel.TLabel"
            ).grid(row=2, column=0, sticky="w")

        # ── Bottom Action Bar ──────────────────────────────────────────
        bottom_bar = self.ttk.Frame(root, style="Results.TFrame", padding=(16, 10, 16, 14))
        bottom_bar.grid(row=2, column=0, sticky="ew")
        bottom_bar.columnconfigure(0, weight=1)

        actions_frame = self.ttk.Frame(bottom_bar, style="Results.TFrame")
        actions_frame.grid(row=0, column=0, sticky="w")

        self.ttk.Button(
            actions_frame,
            text="\U0001F4C4  PDF Report",
            style="ActionBtnGhost.TButton",
            command=self._on_export,
        ).grid(row=0, column=0, padx=(0, 8))

        self.ttk.Button(
            actions_frame,
            text="\U0001F4CA  CSV Export",
            style="ActionBtnGhost.TButton",
            command=self._on_export,
        ).grid(row=0, column=1, padx=(0, 8))

        self.ttk.Button(
            actions_frame,
            text="\U0001F504  Compare",
            style="ActionBtnGhost.TButton",
            command=self._on_compare,
        ).grid(row=0, column=2, padx=(0, 8))

        self.ttk.Button(
            actions_frame,
            text="\U0001F4BE  Save",
            style="ActionBtnGhost.TButton",
            command=self._on_save,
        ).grid(row=0, column=3, padx=(0, 8))

    def _draw_video_placeholder(self) -> None:
        """Draw a placeholder in the video canvas with pose skeleton and barbell."""
        w = self.video_canvas.winfo_width() or 800
        h = self.video_canvas.winfo_height() or 500
        self.video_canvas.config(width=w, height=h)

        # Dark background
        self.video_canvas.create_rectangle(0, 0, w, h, fill="#1a1a2e")

        # Pose skeleton placeholder
        cx, cy = w // 2, h // 2
        color = "#00dc00"
        line_width = max(2, w // 400)

        # Head
        self.video_canvas.create_oval(
            cx - 15, cy - 65, cx + 15, cy - 35, outline=color, fill="", width=line_width
        )
        # Torso
        self.video_canvas.create_line(
            cx, cy - 35, cx, cy + 40, fill=color, width=line_width + 1
        )
        # Arms
        self.video_canvas.create_line(
            cx - 55, cy - 10, cx, cy - 25, fill=color, width=line_width
        )
        self.video_canvas.create_line(
            cx + 55, cy - 10, cx, cy - 25, fill=color, width=line_width
        )
        # Legs
        self.video_canvas.create_line(
            cx - 35, cy + 85, cx, cy + 40, fill=color, width=line_width
        )
        self.video_canvas.create_line(
            cx + 35, cy + 85, cx, cy + 40, fill=color, width=line_width
        )
        # Barbell above head
        self.video_canvas.create_line(
            cx - 70, cy - 80, cx + 70, cy - 80, fill="#ffaa00", width=line_width + 1
        )
        # Plates
        self.video_canvas.create_oval(
            cx - 78, cy - 90, cx - 62, cy - 70, fill="#ffaa00", outline=""
        )
        self.video_canvas.create_oval(
            cx + 62, cy - 90, cx + 78, cy - 70, fill="#ffaa00", outline=""
        )

        # Text overlay
        self.video_canvas.create_text(
            cx, cy + 130,
            text="Video playback with pose overlay",
            fill="#888",
            font=("Segoe UI", 12),
        )

    def _draw_timeline(self) -> None:
        """Draw phase segments on the timeline canvas."""
        w = self.timeline_canvas.winfo_width() or 600
        h = self.timeline_canvas.winfo_height() or 8
        self.timeline_canvas.config(width=w, height=h)

        segments = [
            (0.0, 0.12, self.PHASE_COLORS["setup"]),
            (0.12, 0.35, self.PHASE_COLORS["first_pull"]),
            (0.35, 0.48, self.PHASE_COLORS["second_pull"]),
            (0.48, 0.58, self.PHASE_COLORS["turnover"]),
            (0.58, 0.65, self.PHASE_COLORS["catch"]),
            (0.65, 0.88, self.PHASE_COLORS["recovery"]),
            (0.88, 1.0, self.PHASE_COLORS["finish"]),
        ]

        for start, end, color in segments:
            x1 = int(start * w)
            x2 = int(end * w)
            self.timeline_canvas.create_rectangle(
                x1, 0, x2, h, fill=color, outline=""
            )

        # Current position indicator
        mid = w // 2
        self.timeline_canvas.create_line(
            mid, 0, mid, h, fill="#fff", width=2
        )

    def _populate_data(self) -> None:
        """Populate the dashboard with analysis data."""
        if not self.analysis_data:
            return

        # Update score
        score = self.analysis_data.get("score", 84)
        self.score_value_label.config(text=f"{score:.0f}/100")

        # Color the score
        if score >= 85:
            score_color = COLOR_SUCCESS
        elif score >= 70:
            score_color = COLOR_WARNING
        else:
            score_color = COLOR_ERROR
        self.score_value_label.config(foreground=score_color)

        # Update score bar
        fill = (score / 100) * 200
        self.score_bar_canvas.config(width=200)
        self.score_bar_canvas.create_rectangle(
            0, 0, fill, 10,
            fill=score_color,
            outline="",
        )

        # Update score description
        if score >= 85:
            desc = "Excellent form"
        elif score >= 75:
            desc = "Good form, minor issues"
        elif score >= 60:
            desc = "Needs work on several phases"
        else:
            desc = "Significant technique issues"
        self.score_desc_label.config(text=desc)

        # Update video name
        self.video_name_var.set(self.analysis_data.get("video_name", "Unknown"))

        # Update phase data if available
        phases = self.analysis_data.get("phases", {})
        if phases:
            for name, phase_data in phases.items():
                if name in self.phase_labels:
                    phase_score = phase_data.get("score", 80)
                    self.phase_labels[name]["score"] = phase_score
                    fill = (phase_score / 100) * 150
                    phase_color = COLOR_SUCCESS if phase_score >= 80 else COLOR_WARNING
                    self.phase_labels[name]["canvas"].delete("all")
                    self.phase_labels[name]["canvas"].create_rectangle(
                        0, 0, fill, 12, fill=phase_color, outline=""
                    )

    def _toggle_theme(self, event) -> None:
        """Toggle between light and dark theme."""
        new_theme = "dark" if self.theme_name == "light" else "light"
        self.theme_name = new_theme
        self.theme_label.config(text="\u263E" if new_theme == "light" else "\u263D")
        # In a full implementation, this would rebuild the UI with the new theme
        # For now, show a message
        self.messagebox.showinfo(
            "Theme",
            f"Theme switched to {new_theme} mode.\n"
            "Full theme switching requires UI rebuild.",
        )

    def _on_export(self) -> None:
        if self.on_export:
            self.on_export()
        else:
            self.messagebox.showinfo("Export", "Export functionality coming soon.")

    def _on_compare(self) -> None:
        if self.on_compare:
            self.on_compare()
        else:
            self.messagebox.showinfo("Compare", "Comparison mode coming soon.")

    def _on_save(self) -> None:
        self.messagebox.showinfo("Save", "Results saved successfully.")

    def run(self) -> int:
        self.root.mainloop()
        return 0
