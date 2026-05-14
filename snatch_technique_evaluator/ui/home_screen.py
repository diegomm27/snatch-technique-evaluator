from __future__ import annotations

"""Home / Dashboard screen per WEIA-17 wireframes.

Shows recent analyses, quick stats, and quick actions (New Analysis, Comparison Mode).
"""

import json
from pathlib import Path
from typing import Any

from .design_system import (
    THEME_LIGHT,
    THEME_DARK,
    COLOR_PRIMARY_600,
    COLOR_BORDER_LIGHT,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_ERROR,
)


class RecentAnalysis:
    """Represents a single recent analysis entry."""

    def __init__(self, video_name: str, score: float, time_ago: str, thumbnail: str = "") -> None:
        self.video_name = video_name
        self.score = score
        self.time_ago = time_ago
        self.thumbnail = thumbnail

    @property
    def score_color(self) -> str:
        if self.score is None:
            return COLOR_ERROR
        if self.score >= 85:
            return COLOR_SUCCESS
        if self.score >= 70:
            return COLOR_WARNING
        return COLOR_ERROR


class HomeScreen:
    """Home / Dashboard screen per WEIA-17 wireframes.

    Layout:
    - Top bar with app title, quick actions, and settings
    - Quick stats row (Analyses, Avg Score, Best, This Month)
    - Recent Analyses grid (last 6 videos)
    """

    def __init__(self, on_new_analysis=None, on_comparison=None, on_settings=None, theme: str = "light") -> None:
        self.on_new_analysis = on_new_analysis
        self.on_comparison = on_comparison
        self.on_settings = on_settings
        self.theme_name = theme

        try:
            import ttkbootstrap as ttk
            self._has_ttkb = True
            self.ttk = ttk
        except ImportError:
            import tkinter as tk
            from tkinter import ttk as _ttk
            self._has_ttkb = False
            self.ttk = _ttk

        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
        except ImportError as exc:
            raise RuntimeError("tkinter is unavailable in this environment.") from exc

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox

        if self._has_ttkb:
            self.root = ttk.Window(themename="flatly" if theme == "light" else "darkly")
        else:
            self.root = tk.Tk()

        self.root.title("Weight Lifting Analyzer")
        self.root.geometry("1300x850")
        self.root.minsize(1200, 800)

        theme = THEME_LIGHT if theme == "light" else THEME_DARK
        self.root.configure(bg=theme["bg"])

        self.recent_analyses: list[RecentAnalysis] = []
        self._load_recent()

        self._build_styles(theme)
        self._build_layout(theme)

    def _build_styles(self, theme: dict) -> None:
        if self._has_ttkb:
            style = self.root.style
        else:
            style = self.ttk.Style(self.root)
            try:
                style.theme_use("clam")
            except Exception:
                pass

        # App background
        style.configure("Home.TFrame", background=theme["bg"])

        # Title
        style.configure(
            "HomeTitle.TLabel",
            background=theme["bg"],
            foreground=theme["text_primary"],
            font=("Segoe UI Semibold", 26),
        )

        # Subtitle
        style.configure(
            "HomeSubtitle.TLabel",
            background=theme["bg"],
            foreground=theme["text_secondary"],
            font=("Segoe UI", 12),
        )

        # Stats card
        style.configure(
            "StatCard.TFrame",
            background=theme["surface"],
        )
        style.configure(
            "StatLabel.TLabel",
            background=theme["surface"],
            foreground=theme["text_secondary"],
            font=("Segoe UI", 11),
        )
        style.configure(
            "StatValue.TLabel",
            background=theme["surface"],
            foreground=theme["text_primary"],
            font=("Segoe UI Semibold", 24),
        )

        # Section headers
        style.configure(
            "SectionHeader.TLabel",
            background=theme["bg"],
            foreground=theme["text_primary"],
            font=("Segoe UI Semibold", 18),
        )

        # Recent analysis card
        style.configure(
            "RecentCard.TFrame",
            background=theme["surface"],
        )
        style.configure(
            "RecentVideoName.TLabel",
            background=theme["surface"],
            foreground=theme["text_primary"],
            font=("Segoe UI Semibold", 12),
        )
        style.configure(
            "RecentScore.TLabel",
            background=theme["surface"],
            foreground=theme["text_secondary"],
            font=("Segoe UI", 11),
        )
        style.configure(
            "RecentTime.TLabel",
            background=theme["surface"],
            foreground=theme["text_secondary"],
            font=("Segoe UI", 9),
        )

        # Buttons
        style.configure(
            "HomePrimary.TButton",
            font=("Segoe UI Semibold", 13),
            padding=(16, 10),
            background=theme["primary"],
            foreground="white",
        )
        style.map("HomePrimary.TButton",
                  background=[("active", theme["primary_hover"])])

        style.configure(
            "HomeSecondary.TButton",
            font=("Segoe UI", 12),
            padding=(14, 9),
        )

    def _build_layout(self, theme: dict) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        # ── Top Bar ────────────────────────────────────────────────────
        top_bar = self.ttk.Frame(root, style="Home.TFrame", padding=(24, 16, 24, 12))
        top_bar.grid(row=0, column=0, sticky="ew")
        top_bar.columnconfigure(2, weight=1)

        # Title
        self.ttk.Label(
            top_bar,
            text="Weight Lifting Analyzer",
            style="HomeTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")

        # Quick actions
        actions_frame = self.ttk.Frame(top_bar, style="Home.TFrame")
        actions_frame.grid(row=0, column=2, sticky="e")

        self.ttk.Button(
            actions_frame,
            text="+ New Analysis",
            style="HomePrimary.TButton",
            command=self._on_new_analysis,
        ).grid(row=0, column=0, padx=(0, 10))

        self.ttk.Button(
            actions_frame,
            text="Compare",
            style="HomeSecondary.TButton",
            command=self._on_comparison,
        ).grid(row=0, column=1, padx=(0, 8))

        self.ttk.Button(
            actions_frame,
            text="\u2699",
            style="HomeSecondary.TButton",
            command=self._on_settings,
        ).grid(row=0, column=2)

        # Subtitle
        self.ttk.Label(
            top_bar,
            text="Desktop application for analyzing weightlifting snatch technique using pose estimation",
            style="HomeSubtitle.TLabel",
            wraplength=1100,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # ── Content Area ───────────────────────────────────────────────
        content = self.ttk.Frame(root, style="Home.TFrame", padding=(24, 12, 24, 20))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=0)
        content.rowconfigure(1, weight=1)

        # ── Quick Stats Row ────────────────────────────────────────────
        stats_frame = self.ttk.Frame(content, style="Home.TFrame")
        stats_frame.grid(row=0, column=0, sticky="ew")
        stats_frame.columnconfigure((0, 1, 2, 3), weight=1)

        stats = self._compute_stats()
        stat_labels = ["Analyses", "Avg Score", "Best", "This Month"]
        for i, (label, value) in enumerate(zip(stat_labels, stats)):
            card = self.ttk.Frame(stats_frame, style="StatCard.TFrame", padding=(20, 16))
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 12, 0))

            self.ttk.Label(
                card, text=label, style="StatLabel.TLabel"
            ).grid(row=0, column=0, sticky="w")

            self.ttk.Label(
                card, text=str(value), style="StatValue.TLabel"
            ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        # ── Recent Analyses Section ────────────────────────────────────
        recent_section = self.ttk.Frame(content, style="Home.TFrame")
        recent_section.grid(row=1, column=0, sticky="nsew", pady=(20, 0))
        recent_section.columnconfigure(0, weight=1)
        recent_section.rowconfigure(0, weight=0)
        recent_section.rowconfigure(1, weight=1)

        self.ttk.Label(
            recent_section,
            text="Recent Analyses",
            style="SectionHeader.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        # Grid of recent analysis cards
        grid_frame = self.ttk.Frame(recent_section, style="Home.TFrame")
        grid_frame.grid(row=1, column=0, sticky="nsew")
        grid_frame.columnconfigure(tuple(range(3)), weight=1)

        # Populate with recent analyses (up to 6, in 2 rows of 3)
        for idx, analysis in enumerate(self.recent_analyses[:6]):
            row = idx // 3
            col = idx % 3
            card = self._create_recent_card(grid_frame, analysis)
            card.grid(row=row, column=col, sticky="nsew", padx=(0, 12) if col < 2 else (0, 0), pady=(0, 12) if row == 0 else (0, 0))

        # Handle empty state
        if not self.recent_analyses:
            empty_card = self.ttk.Frame(grid_frame, style="StatCard.TFrame", padding=(40, 30))
            empty_card.grid(row=0, column=0, columnspan=3, sticky="nsew")
            self.ttk.Label(
                empty_card,
                text="No analyses yet.\nStart by importing a snatch video.",
                style="HomeSubtitle.TLabel",
                justify="center",
            ).pack()

    def _create_recent_card(self, parent: Any, analysis: RecentAnalysis) -> Any:
        """Create a single recent analysis card widget."""
        card = self.ttk.Frame(parent, style="RecentCard.TFrame", padding=(16, 14))
        card.columnconfigure(0, weight=1)
        score_value = analysis.score if analysis.score is not None else 0

        # Score badge at top
        score_color = analysis.score_color
        self.ttk.Label(
            card,
            text=f"{score_value:.0f}/100",
            style="RecentScore.TLabel",
        ).grid(row=0, column=0, sticky="e")

        # Video name
        self.ttk.Label(
            card,
            text=analysis.video_name,
            style="RecentVideoName.TLabel",
            wraplength=300,
        ).grid(row=1, column=0, sticky="w", pady=(8, 2))

        # Time ago
        self.ttk.Label(
            card,
            text=analysis.time_ago,
            style="RecentTime.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(2, 0))

        # Score bar indicator
        bar_frame = self.ttk.Frame(card, style="RecentCard.TFrame")
        bar_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        bar_frame.columnconfigure(0, weight=1)

        bar = self.tk.Canvas(bar_frame, width=200, height=6, bg=COLOR_BORDER_LIGHT, highlightthickness=0)
        bar.grid(row=0, column=0, sticky="ew")
        fill_width = (score_value / 100) * 200
        bar.create_rectangle(0, 0, fill_width, 6, fill=score_color, outline="")

        # Clickable - bind to open analysis
        card.bind("<Button-1>", lambda e: self._on_open_analysis(analysis))

        return card

    def _compute_stats(self) -> list:
        """Compute quick stats from recent analyses."""
        if not self.recent_analyses:
            return ["0", "—", "—", "0"]

        scores = [a.score for a in self.recent_analyses if a.score is not None]
        if not scores:
            return [str(len(self.recent_analyses)), "â€”", "â€”", "0"]
        avg_score = sum(scores) / len(scores)
        best_score = max(scores)
        return [
            str(len(self.recent_analyses)),
            f"{avg_score:.0f}",
            f"{best_score:.0f}",
            str(len([a for a in self.recent_analyses if "today" in a.time_ago.lower() or "hours" in a.time_ago.lower()])),
        ]

    def _load_recent(self) -> None:
        """Load recent analyses from disk or use sample data."""
        # Look for output directories
        outputs_dir = Path(__file__).resolve().parent.parent.parent / "outputs"
        if outputs_dir.exists():
            for item in sorted(outputs_dir.iterdir(), reverse=True)[:6]:
                if item.is_dir():
                    metrics_file = item / "metrics.json"
                    if metrics_file.exists():
                        try:
                            data = json.loads(metrics_file.read_text())
                            score = data.get("score", 0)
                            video_name = item.name
                            self.recent_analyses.append(RecentAnalysis(
                                video_name=video_name,
                                score=score,
                                time_ago="Recently",
                            ))
                        except (json.JSONDecodeError, KeyError):
                            self.recent_analyses.append(RecentAnalysis(
                                video_name=item.name,
                                score=0,
                                time_ago="Recently",
                            ))
                    else:
                        self.recent_analyses.append(RecentAnalysis(
                            video_name=item.name,
                            score=0,
                            time_ago="Recently",
                        ))

    def _on_new_analysis(self) -> None:
        if self.on_new_analysis:
            self.on_new_analysis()
        else:
            self.messagebox.showinfo("New Analysis", "Select a video to begin analysis.")

    def _on_comparison(self) -> None:
        if self.on_comparison:
            self.on_comparison()
        else:
            self.messagebox.showinfo("Comparison Mode", "Select 2+ videos to compare.")

    def _on_settings(self) -> None:
        if self.on_settings:
            self.on_settings()
        else:
            self.messagebox.showinfo("Settings", "Settings panel coming soon.")

    def _on_open_analysis(self, analysis: RecentAnalysis) -> None:
        """Open a specific analysis result."""
        self.messagebox.showinfo(
            "Analysis Result",
            f"Opening: {analysis.video_name}\nScore: {analysis.score:.0f}/100",
        )

    def run(self) -> int:
        self.root.mainloop()
        return 0
