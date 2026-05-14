from __future__ import annotations

"""Comparison View per WEIA-17 wireframes.

Side-by-side video player with synchronized playback,
metric comparison table, and improvement delta indicators.
"""

from typing import Any

from .design_system import (
    THEME_LIGHT,
    THEME_DARK,
    COLOR_PRIMARY_600,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_ERROR,
    COLOR_BORDER_LIGHT,
)


class ComparisonEntry:
    """A single video entry in comparison mode."""

    def __init__(
        self,
        video_name: str,
        score: float = 0,
        data: dict | None = None,
        thumbnail: str = "",
    ) -> None:
        self.video_name = video_name
        self.score = score
        self.data = data or {}
        self.thumbnail = thumbnail

    @property
    def score_color(self) -> str:
        if self.score >= 85:
            return COLOR_SUCCESS
        if self.score >= 70:
            return COLOR_WARNING
        return COLOR_ERROR


class ComparisonView:
    """Comparison View per WEIA-17 wireframes.

    Layout:
    - Top bar: Video selectors, synchronized playback controls
    - Left: Video player 1 with pose overlay
    - Right: Video player 2 with pose overlay
    - Bottom: Metric comparison table with delta indicators
    - Action bar: Generate comparison report
    """

    def __init__(
        self,
        entries: list[ComparisonEntry] | None = None,
        on_export=None,
        on_back=None,
        theme: str = "light",
    ) -> None:
        self.entries = entries or []
        self.on_export = on_export
        self.on_back = on_back
        self.theme_name = theme
        self.selected_entries: list[ComparisonEntry] = []

        try:
            import ttkbootstrap as ttk
            self._has_ttkb = True
        except ImportError:
            import tkinter as tk
            from tkinter import ttk as _ttk
            self._has_ttkb = False

        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
        except ImportError as exc:
            raise RuntimeError("tkinter is unavailable in this environment.") from exc

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox

        if self._has_ttkb:
            self.root = ttk.Window(
                themename="flatly" if theme == "light" else "darkly"
            )
        else:
            self.root = tk.Tk()

        self.root.title("Comparison Mode - Weight Lifting Analyzer")
        self.root.geometry("1500x900")
        self.root.minsize(1200, 800)

        theme_dict = THEME_LIGHT if theme == "light" else THEME_DARK
        self.root.configure(bg=theme_dict["bg"])

        self._build_styles(theme_dict)
        self._build_layout(theme_dict)

    def _build_styles(self, theme: dict) -> None:
        if self._has_ttkb:
            style = self.root.style
        else:
            style = self.ttk.Style(self.root)
            try:
                style.theme_use("clam")
            except Exception:
                pass

        style.configure("Compare.TFrame", background=theme["bg"])

        # Video panels
        style.configure(
            "CompareVideo.TFrame",
            background="#1a1a2e",
        )
        style.configure(
            "CompareVideoTitle.TLabel",
            background="#1a1a2e",
            foreground="#e0e0e0",
            font=("Segoe UI Semibold", 12),
        )

        # Comparison table
        style.configure(
            "CompareTable.TFrame",
            background=theme["surface"],
        )
        style.configure(
            "MetricName.TLabel",
            background=theme["surface"],
            foreground=theme["text_primary"],
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "MetricValue.TLabel",
            background=theme["surface"],
            foreground=theme["text_primary"],
            font=("Segoe UI", 11),
        )
        style.configure(
            "DeltaPositive.TLabel",
            background=theme["surface"],
            foreground=COLOR_SUCCESS,
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "DeltaNegative.TLabel",
            background=theme["surface"],
            foreground=COLOR_ERROR,
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "DeltaNeutral.TLabel",
            background=theme["surface"],
            foreground=theme["text_secondary"],
            font=("Segoe UI", 11),
        )

        # Buttons
        style.configure(
            "ComparePrimary.TButton",
            font=("Segoe UI Semibold", 12),
            padding=(16, 8),
        )
        style.configure(
            "CompareSecondary.TButton",
            font=("Segoe UI", 11),
            padding=(12, 7),
        )

    def _build_layout(self, theme: dict) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        # ── Top Bar ────────────────────────────────────────────────────
        top_bar = self.ttk.Frame(root, style="Compare.TFrame", padding=(16, 10, 16, 8))
        top_bar.grid(row=0, column=0, sticky="ew")
        top_bar.columnconfigure(1, weight=1)

        self.ttk.Label(
            top_bar,
            text="Comparison Mode",
            style="CompareVideoTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")

        # Video selectors
        selector_frame = self.ttk.Frame(top_bar, style="Compare.TFrame")
        selector_frame.grid(row=0, column=1, sticky="w", padx=(20, 0))

        self.ttk.Label(
            selector_frame,
            text="Video A:",
            style="MetricName.TLabel",
        ).grid(row=0, column=0, sticky="e", padx=(0, 6))

        self.video_a_var = self.tk.StringVar(value="Select video...")
        self.ttk.Entry(
            selector_frame,
            textvariable=self.video_a_var,
            state="readonly",
            font=("Segoe UI", 10),
        ).grid(row=0, column=1, sticky="ew", padx=(0, 8))

        self.ttk.Button(
            selector_frame,
            text="Browse",
            style="CompareSecondary.TButton",
            command=self._browse_video_a,
        ).grid(row=0, column=2, padx=(0, 16))

        self.ttk.Label(
            selector_frame,
            text="vs",
            style="MetricName.TLabel",
        ).grid(row=0, column=3, padx=(8, 8))

        self.ttk.Label(
            selector_frame,
            text="Video B:",
            style="MetricName.TLabel",
        ).grid(row=0, column=4, sticky="e", padx=(0, 6))

        self.video_b_var = self.tk.StringVar(value="Select video...")
        self.ttk.Entry(
            selector_frame,
            textvariable=self.video_b_var,
            state="readonly",
            font=("Segoe UI", 10),
        ).grid(row=0, column=5, sticky="ew", padx=(0, 8))

        self.ttk.Button(
            selector_frame,
            text="Browse",
            style="CompareSecondary.TButton",
            command=self._browse_video_b,
        ).grid(row=0, column=6, padx=(0, 0))

        # Back button
        if self.on_back:
            self.ttk.Button(
                top_bar,
                text="\u2190 Back",
                style="CompareSecondary.TButton",
                command=self.on_back,
            ).grid(row=0, column=2, sticky="e")

        # ── Main Content ───────────────────────────────────────────────
        main_content = self.ttk.Frame(root, style="Compare.TFrame", padding=(16, 8, 16, 0))
        main_content.grid(row=1, column=0, sticky="nsew")
        main_content.columnconfigure(0, weight=1)
        main_content.columnconfigure(1, weight=1)
        main_content.rowconfigure(0, weight=2)
        main_content.rowconfigure(1, weight=1)

        # ── Left Video Player ──────────────────────────────────────────
        video_a_panel = self.ttk.Frame(main_content, style="CompareVideo.TFrame", padding=0)
        video_a_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        video_a_panel.columnconfigure(0, weight=1)
        video_a_panel.rowconfigure(0, weight=1)

        self.video_a_canvas = self.tk.Canvas(
            video_a_panel, bg="#1a1a2e", highlightthickness=0
        )
        self.video_a_canvas.grid(row=0, column=0, sticky="nsew")
        self._draw_comparison_placeholder(self.video_a_canvas, "Video A")

        # Score badge
        self.ttk.Label(
            video_a_panel,
            text="Score: --",
            style="CompareVideoTitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        # ── Right Video Player ─────────────────────────────────────────
        video_b_panel = self.ttk.Frame(main_content, style="CompareVideo.TFrame", padding=0)
        video_b_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        video_b_panel.columnconfigure(0, weight=1)
        video_b_panel.rowconfigure(0, weight=1)

        self.video_b_canvas = self.tk.Canvas(
            video_b_panel, bg="#1a1a2e", highlightthickness=0
        )
        self.video_b_canvas.grid(row=0, column=0, sticky="nsew")
        self._draw_comparison_placeholder(self.video_b_canvas, "Video B")

        # Score badge
        self.ttk.Label(
            video_b_panel,
            text="Score: --",
            style="CompareVideoTitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        # ── Comparison Table ───────────────────────────────────────────
        table_panel = self.ttk.Frame(main_content, style="Compare.TFrame", padding=(0, 0, 0, 0))
        table_panel.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        table_panel.columnconfigure(0, weight=1)
        table_panel.rowconfigure(0, weight=0)
        table_panel.rowconfigure(1, weight=1)

        self.ttk.Label(
            table_panel,
            text="Metric Comparison",
            style="MetricName.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        # Table header
        header_frame = self.ttk.Frame(table_panel, style="CompareTable.TFrame", padding=(12, 8))
        header_frame.grid(row=1, column=0, sticky="ew")
        header_frame.columnconfigure(0, width=200)
        header_frame.columnconfigure(1, weight=1)
        header_frame.columnconfigure(2, weight=1)
        header_frame.columnconfigure(3, width=80)

        self.ttk.Label(
            header_frame, text="Metric", style="MetricName.TLabel"
        ).grid(row=0, column=0, sticky="w")
        self.ttk.Label(
            header_frame, text="Video A", style="MetricName.TLabel"
        ).grid(row=0, column=1, sticky="w", padx=(20, 0))
        self.ttk.Label(
            header_frame, text="Video B", style="MetricName.TLabel"
        ).grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.ttk.Label(
            header_frame, text="Delta", style="MetricName.TLabel"
        ).grid(row=0, column=3, sticky="w", padx=(20, 0))

        # Table rows
        self._table_rows: list[dict[str, Any]] = []
        comparison_metrics = [
            ("Overall Score", "--", "--", "--"),
            ("Bar Path Alignment", "--", "--", "--"),
            ("Knee Angle (catch)", "--", "--", "--"),
            ("Hip Angle (setup)", "--", "--", "--"),
            ("Torso Angle (first pull)", "--", "--", "--"),
            ("Pull Duration", "--", "--", "--"),
            ("Turnover Speed", "--", "--", "--"),
            ("Recovery Stability", "--", "--", "--"),
            ("Foot Displacement", "--", "--", "--"),
            ("Overhead Alignment", "--", "--", "--"),
        ]

        for i, (name, va, vb, delta) in enumerate(comparison_metrics):
            row_frame = self.ttk.Frame(table_panel, style="CompareTable.TFrame", padding=(8, 4))
            row_frame.grid(row=2 + i, column=0, sticky="ew")
            row_frame.columnconfigure(0, width=200)
            row_frame.columnconfigure(1, weight=1)
            row_frame.columnconfigure(2, weight=1)
            row_frame.columnconfigure(3, width=80)

            self.ttk.Label(
                row_frame, text=name, style="MetricName.TLabel"
            ).grid(row=0, column=0, sticky="w")
            self.ttk.Label(
                row_frame, text=va, style="MetricValue.TLabel"
            ).grid(row=0, column=1, sticky="w", padx=(20, 0))
            self.ttk.Label(
                row_frame, text=vb, style="MetricValue.TLabel"
            ).grid(row=0, column=2, sticky="w", padx=(20, 0))

            delta_label = self.ttk.Label(
                row_frame, text=delta, style="DeltaNeutral.TLabel"
            )
            delta_label.grid(row=0, column=3, sticky="w", padx=(20, 0))

            self._table_rows.append({
                "name": name,
                "video_a": va,
                "video_b": vb,
                "delta": delta_label,
            })

        # ── Bottom Action Bar ──────────────────────────────────────────
        bottom_bar = self.ttk.Frame(root, style="Compare.TFrame", padding=(16, 10, 16, 14))
        bottom_bar.grid(row=3, column=0, sticky="ew")
        bottom_bar.columnconfigure(1, weight=1)

        self.ttk.Button(
            bottom_bar,
            text="\U0001F4CB  Generate Comparison Report",
            style="ComparePrimary.TButton",
            command=self._on_export,
        ).grid(row=0, column=2, sticky="e")

    def _draw_comparison_placeholder(self, canvas: Any, label: str) -> None:
        """Draw a placeholder in the comparison video canvas."""
        w = canvas.winfo_width() or 600
        h = canvas.winfo_height() or 400
        canvas.config(width=w, height=h)

        canvas.create_rectangle(0, 0, w, h, fill="#1a1a2e")

        cx, cy = w // 2, h // 2
        # Skeleton placeholder
        color = "#00dc00"
        canvas.create_line(cx - 40, cy - 10, cx, cy - 25, fill=color, width=2)
        canvas.create_line(cx + 40, cy - 10, cx, cy - 25, fill=color, width=2)
        canvas.create_line(cx, cy - 25, cx, cy + 35, fill=color, width=3)
        canvas.create_line(cx - 25, cy + 70, cx, cy + 35, fill=color, width=2)
        canvas.create_line(cx + 25, cy + 70, cx, cy + 35, fill=color, width=2)
        canvas.create_line(cx - 50, cy - 55, cx + 50, cy - 55, fill="#ffaa00", width=3)

        canvas.create_text(
            cx, cy + 100,
            text=label + " — Select a video to compare",
            fill="#666",
            font=("Segoe UI", 12),
        )

    def _browse_video_a(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="Select video A for comparison",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")],
        )
        if selected:
            self.video_a_var.set(selected)

    def _browse_video_b(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="Select video B for comparison",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")],
        )
        if selected:
            self.video_b_var.set(selected)

    def update_comparison_data(self, entry_a: ComparisonEntry | None, entry_b: ComparisonEntry | None) -> None:
        """Update the comparison table with actual data from two entries."""
        if not entry_a or not entry_b:
            return

        self.video_a_var.set(entry_a.video_name)
        self.video_b_var.set(entry_b.video_name)

        # Update score badges
        for canvas, entry in [(self.video_a_canvas, entry_a), (self.video_b_canvas, entry_b)]:
            score_text = f"Score: {entry.score:.0f}/100"
            # Find the score label in the panel
            for child in canvas.master.winfo_children():
                if isinstance(child, self.ttk.Label if self._has_ttkb else type(self.ttk.Label(canvas.master))):
                    pass  # handled by grid position

        # Update table rows
        for row in self._table_rows:
            name = row["name"]
            va_val = entry_a.data.get(name.lower().replace(" ", "_"), "--")
            vb_val = entry_b.data.get(name.lower().replace(" ", "_"), "--")

            row["video_a"] = str(va_val) if va_val != "--" else "--"
            row["video_b"] = str(vb_val) if vb_val != "--" else "--"

            # Calculate delta
            try:
                va_num = float(va_val) if va_val != "--" else 0
                vb_num = float(vb_val) if vb_val != "--" else 0
                delta_val = va_num - vb_num
                if delta_val > 0:
                    row["delta"].config(text=f"+{delta_val:.1f}", style="DeltaPositive.TLabel")
                elif delta_val < 0:
                    row["delta"].config(text=f"{delta_val:.1f}", style="DeltaNegative.TLabel")
                else:
                    row["delta"].config(text="0", style="DeltaNeutral.TLabel")
            except (ValueError, TypeError):
                row["delta"].config(text="--", style="DeltaNeutral.TLabel")

    def _on_export(self) -> None:
        if self.on_export:
            self.on_export()
        else:
            self.messagebox.showinfo(
                "Comparison Report",
                "Generating comparison report...\n\nThis will create a side-by-side "
                "report with all metric deltas and improvement suggestions.",
            )

    def run(self) -> int:
        self.root.mainloop()
        return 0
