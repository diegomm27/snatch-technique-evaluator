from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


def _preferred_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        # Verify the GPU is actually usable (catches sm_120 / driver mismatches)
        torch.zeros(1, device="cuda:0")
        return "cuda:0"
    except Exception:
        return "cpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snatch Detector V1")
    parser.set_defaults(theme="light")
    subparsers = parser.add_subparsers(dest="mode")

    analyze_parser = subparsers.add_parser("analyze", help="Track the bar path in a snatch attempt.")
    launch_parser = subparsers.add_parser("launch", help="Open the desktop launcher UI.")
    dashboard_parser = subparsers.add_parser("dashboard", help="Open the home dashboard UI.")

    for target in (parser, analyze_parser, launch_parser, dashboard_parser):
        _add_common_analysis_args(target)

    launch_parser.add_argument(
        "--theme",
        default="light",
        choices=["light", "dark"],
        help="UI theme.",
    )

    args = parser.parse_args(argv)
    if args.mode is None:
        args.mode = "launch"
    return args


def _add_common_analysis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional path to a video. If omitted, a file picker opens.",
    )
    parser.add_argument(
        "--backend",
        default="yolo",
        choices=["yolo"],
        help="Pose backend to use.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="Pose model name or path.",
    )
    parser.add_argument(
        "--device",
        default=_preferred_device(),
        help="Torch device, for example auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--auto-pause",
        action="store_true",
        help="Pause playback automatically when the lift completes.",
    )
    parser.add_argument(
        "--pause-mode",
        default="after-recovery-stable",
        choices=["after-recovery-stable"],
        help="Auto-pause trigger mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory root.",
    )


def pick_video() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("tkinter is unavailable in this environment.") from exc

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title="Select a side-view snatch video",
        filetypes=[
            ("Video files", "*.mp4 *.mov *.avi *.mkv"),
            ("MP4", "*.mp4"),
            ("MOV", "*.mov"),
            ("AVI", "*.avi"),
            ("MKV", "*.mkv"),
        ],
    )
    root.destroy()
    if not selected:
        return None
    return Path(selected)


def _load_analysis_symbols():
    from .analysis import (
        AnalyzerConfig,
        SnatchAnalysisSession,
    )

    return AnalyzerConfig, SnatchAnalysisSession


def _model_for_backend(model_text: str) -> str:
    return model_text.strip() or "yolo11n-pose.pt"


class SnatchLauncher:
    """Main launcher with full UI integration.

    Screens:
    1. Home Dashboard - recent analyses, quick stats, quick actions
    2. Analysis Configuration - presets, parameters, advanced settings
    3. Results Dashboard - 3-zone layout (video + metrics + actions)
    4. Comparison View - side-by-side video comparison
    """

    def __init__(self, model: str, theme: str = "light") -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
        except ImportError as exc:
            raise RuntimeError("tkinter is unavailable in this environment.") from exc
        try:
            import ttkbootstrap as ttk
        except ImportError:
            from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.model_default = model
        self.theme_name = theme
        self.worker: threading.Thread | None = None
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()

        if hasattr(ttk, "Window"):
            self.root = ttk.Window(themename="flatly" if theme == "light" else "darkly")
        else:
            self.root = tk.Tk()
        self.root.title("Weight Lifting Analyzer")
        self.root.geometry("1300x850")
        self.root.minsize(1200, 800)
        self.root.configure(bg="#f6f7f9")
        self._build_styles()
        self._build_layout()
        self.root.after(150, self._poll_worker)

    def _build_styles(self) -> None:
        if hasattr(self.ttk, "Window"):
            style = self.root.style
        else:
            style = self.ttk.Style(self.root)
            try:
                style.theme_use("clam")
            except Exception:
                pass
        style.configure("App.TFrame", background="#f6f7f9")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure(
            "Title.TLabel",
            background="#f6f7f9",
            foreground="#171b22",
            font=("Segoe UI Semibold", 24),
        )
        style.configure(
            "Body.TLabel",
            background="#f6f7f9",
            foreground="#4a5565",
            font=("Segoe UI", 10),
        )
        style.configure(
            "CardTitle.TLabel",
            background="#ffffff",
            foreground="#171b22",
            font=("Segoe UI Semibold", 13),
        )
        style.configure(
            "CardBody.TLabel",
            background="#ffffff",
            foreground="#4a5565",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Accent.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(10, 8),
        )
        style.configure("Secondary.TButton", font=("Segoe UI", 10))
        style.configure(
            "Switch.TCheckbutton",
            background="#ffffff",
            foreground="#171b22",
            font=("Segoe UI", 10),
        )

    def _build_layout(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        # Header
        header = self.ttk.Frame(root, style="App.TFrame", padding=(26, 22, 26, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.ttk.Label(header, text="Weight Lifting Analyzer", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.ttk.Label(
            header,
            text="Desktop application for analyzing weightlifting snatch technique using pose estimation",
            style="Body.TLabel",
            wraplength=800,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        # Content
        content = self.ttk.Frame(root, style="App.TFrame", padding=(26, 8, 26, 18))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        # Quick actions row
        actions_frame = self.ttk.Frame(content, style="App.TFrame")
        actions_frame.grid(row=0, column=0, sticky="nsew")
        actions_frame.columnconfigure(tuple(range(4)), weight=1)

        self.ttk.Button(
            actions_frame,
            text="+ New Analysis",
            style="Accent.TButton",
            command=self._open_home_screen,
        ).grid(row=0, column=0, padx=(0, 8), sticky="ew")

        self.ttk.Button(
            actions_frame,
            text="Analysis Config",
            style="Secondary.TButton",
            command=self._open_analysis_config,
        ).grid(row=0, column=1, padx=(0, 8), sticky="ew")

        self.ttk.Button(
            actions_frame,
            text="Compare",
            style="Secondary.TButton",
            command=self._open_comparison,
        ).grid(row=0, column=2, padx=(0, 8), sticky="ew")

        self.ttk.Button(
            actions_frame,
            text="Settings",
            style="Secondary.TButton",
            command=self._open_settings,
        ).grid(row=0, column=3, sticky="ew")

        # Info card
        info_card = self.ttk.Frame(content, style="Card.TFrame", padding=(20, 16))
        info_card.grid(row=1, column=0, sticky="nsew", pady=(16, 0))
        info_card.columnconfigure(0, weight=1)

        self.ttk.Label(
            info_card,
            text="Getting Started",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")

        steps = [
            "1. Click '+ New Analysis' to select a video and run the full analysis pipeline.",
            "2. Click 'Analysis Config' to configure presets, parameters, and advanced settings.",
            "3. Click 'Compare' to view side-by-side comparison of two snatch attempts.",
            "4. After analysis, the Results Dashboard shows the 3-zone layout: video playback "
            "with pose overlay, phase-by-phase metrics, and export actions.",
        ]
        step_text = "\n\n".join(steps)
        self.ttk.Label(
            info_card,
            text=step_text,
            style="CardBody.TLabel",
            wraplength=700,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        # Log area
        self.log_text = self.tk.Text(
            content,
            height=8,
            bg="#211915",
            fg="#f7efe2",
            insertbackground="#f7efe2",
            relief="flat",
            font=("Consolas", 10),
            wrap="word",
        )
        self.log_text.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        self.log_text.insert("1.0", "Launcher ready.\n")
        self.log_text.configure(state="disabled")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _open_home_screen(self) -> None:
        """Open the home dashboard screen."""
        try:
            from .ui.home_screen import HomeScreen
        except ImportError as exc:
            self.messagebox.showerror("Import Error", f"Failed to import HomeScreen: {exc}")
            return

        def on_new_analysis() -> None:
            self._open_analysis_config()

        def on_comparison() -> None:
            self._open_comparison()

        def on_settings() -> None:
            self.messagebox.showinfo("Settings", "Settings panel coming soon.")

        self._append_log("Opening Home Dashboard...")
        home = HomeScreen(
            on_new_analysis=on_new_analysis,
            on_comparison=on_comparison,
            on_settings=on_settings,
            theme=self.theme_name,
        )
        home.run()

    def _open_analysis_config(self) -> None:
        """Open the analysis configuration screen."""
        try:
            from .ui.analysis_config import AnalysisConfiguration
        except ImportError as exc:
            self.messagebox.showerror("Import Error", f"Failed to import AnalysisConfiguration: {exc}")
            return

        def on_run_analysis(**kwargs: Any) -> None:
            self._append_log(f"Running analysis with preset: {kwargs.get('preset', 'unknown')}")
            self._run_analysis(
                video_path=None,
                pose_model=kwargs.get("pose_model", "yolo11n-pose.pt"),
                sensitivity=kwargs.get("sensitivity", 0.7),
                auto_pause=kwargs.get("auto_pause", True),
                reference=kwargs.get("reference", "default_reference.json"),
                output_format=kwargs.get("output_format", "pdf"),
                device=kwargs.get("device", "auto"),
            )

        def on_back() -> None:
            self._append_log("Returning to launcher.")

        self._append_log("Opening Analysis Configuration...")
        config = AnalysisConfiguration(
            on_run_analysis=on_run_analysis,
            on_back=on_back,
            theme=self.theme_name,
        )
        config.run()

    def _open_comparison(self) -> None:
        """Open the comparison view screen."""
        try:
            from .ui.comparison_view import ComparisonView
        except ImportError as exc:
            self.messagebox.showerror("Import Error", f"Failed to import ComparisonView: {exc}")
            return

        def on_export() -> None:
            self.messagebox.showinfo(
                "Comparison Report",
                "Generating comparison report...\n\nThis will create a side-by-side "
                "report with all metric deltas and improvement suggestions.",
            )

        def on_back() -> None:
            self._append_log("Returning to launcher.")

        self._append_log("Opening Comparison Mode...")
        comparison = ComparisonView(
            on_export=on_export,
            on_back=on_back,
            theme=self.theme_name,
        )
        comparison.run()

    def _open_settings(self) -> None:
        self.messagebox.showinfo(
            "Settings",
            "Settings panel coming soon.\n\n"
            "Available settings:\n"
            "- Theme (light/dark)\n"
            "- Default preset profile\n"
            "- Output directory\n"
            "- Pose model selection\n"
            "- Language / i18n",
        )

    def _run_analysis(
        self,
        video_path: Path | None = None,
        pose_model: str = "yolo11n-pose.pt",
        sensitivity: float = 0.7,
        auto_pause: bool = True,
        reference: str = "default_reference.json",
        output_format: str = "pdf",
        device: str = "auto",
    ) -> None:
        """Run the analysis pipeline."""
        if video_path is None:
            video_path = pick_video()
        if video_path is None:
            self.messagebox.showerror("Missing Video", "Select a snatch attempt video first.")
            return
        if not video_path.exists():
            self.messagebox.showerror("Video Not Found", f"Could not find:\n{video_path}")
            return

        self._append_log(f"Starting analysis: {video_path.name}")

        def _run() -> None:
            try:
                AnalyzerConfig, SnatchAnalysisSession = _load_analysis_symbols()
                session = SnatchAnalysisSession(
                    AnalyzerConfig(
                        video_path=video_path,
                        pose_backend_name="yolo",
                        model_name=_model_for_backend(pose_model),
                        device=_preferred_device() if device == "auto" else device,
                        reference_path=None,
                        auto_pause=auto_pause,
                        pause_mode="after-recovery-stable",
                        interactive=True,
                        show_live_window=True,
                        persist_outputs=True,
                    )
                )
                summary = session.run()
                self.events.put(
                    ("success", {"kind": "analysis", "output_dir": str(session.output_dir), "summary": summary})
                )
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self.worker = threading.Thread(target=_run, daemon=True)
        self.worker.start()

    def _run_task(self, title: str, task, kwargs: dict, on_success=None) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.messagebox.showinfo("Busy", "Wait for the current run to finish.")
            return

        self._append_log(title)

        def runner() -> None:
            try:
                payload = task(**kwargs)
                self.events.put(("success", payload))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self.worker = threading.Thread(target=runner, daemon=True)
        self.worker.start()
        self._on_success = on_success

    def _poll_worker(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "success":
                    message = self._handle_success_payload(payload)
                    self._append_log(message)
                    callback = getattr(self, "_on_success", None)
                    self._on_success = None
                    if callable(callback):
                        self.root.after(100, callback)
                else:
                    self._append_log("Error: " + payload)
                    self.messagebox.showerror("Snatch Detector", payload)
                    self._on_success = None
        except queue.Empty:
            pass
        self.root.after(150, self._poll_worker)

    def _handle_success_payload(self, payload: dict[str, Any]) -> str:
        kind = payload.get("kind")
        if kind == "analysis":
            summary = payload["summary"]
            self._render_analysis_report(summary)
            lines = [
                f"Tracking complete: {payload['output_dir']}",
                f"Warnings: {len(summary.get('warnings', []))}",
            ]
            if summary.get("completion_frame") is not None:
                lines.append(f"Completion frame: {summary['completion_frame']}")
            else:
                lines.append("Completion frame: not detected")
            return "\n".join(lines)
        return str(payload)

    def _render_analysis_report(self, summary: dict[str, Any]) -> None:
        video_name = summary.get("annotated_video_filename") or "annotated.mp4"
        report_lines = [
            f"Annotated video: {video_name}",
            "Frame data: frames.csv",
            "Metrics: metrics.json",
        ]
        findings = summary.get("warnings", [])
        if findings:
            report_lines.append("")
            report_lines.append("Warnings:")
            for finding in findings[:3]:
                report_lines.append(f"- {finding}")
        if summary.get("completion_frame") is not None:
            report_lines.append("")
            report_lines.append(f"Completion frame: {summary['completion_frame']}")

        self._append_log("Outputs:")
        for line in report_lines:
            self._append_log(line)

    def run(self) -> int:
        self.root.mainloop()
        return 0


class EasySnatchLauncher:
    """Single-window workflow for the common desktop analysis path."""

    MODEL_CHOICES = ("yolo11n-pose.pt", "yolo11x-pose.pt")
    DEVICE_CHOICES = ("auto", "cpu", "cuda", "cuda:0")

    def __init__(
        self,
        model: str,
        theme: str = "light",
        video_path: Path | None = None,
        device: str = "auto",
    ) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
        except ImportError as exc:
            raise RuntimeError("tkinter is unavailable in this environment.") from exc
        try:
            import ttkbootstrap as ttk
        except ImportError:
            from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.theme_name = theme
        self.worker: threading.Thread | None = None
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.selected_video = video_path
        self.last_output_dir: Path | None = None
        self.recent_outputs: list[Path] = []
        self.model_choices = self.MODEL_CHOICES if model in self.MODEL_CHOICES else (*self.MODEL_CHOICES, model)
        self.device_choices = self.DEVICE_CHOICES if device in self.DEVICE_CHOICES else (*self.DEVICE_CHOICES, device)

        if hasattr(ttk, "Window"):
            self.root = ttk.Window(themename="flatly" if theme == "light" else "darkly")
        else:
            self.root = tk.Tk()
        self.root.title("Snatch Technique Evaluator")
        self.root.geometry("1180x760")
        self.root.minsize(980, 680)

        self.bg = "#f5f7fb" if theme == "light" else "#111827"
        self.surface = "#ffffff" if theme == "light" else "#1f2937"
        self.text = "#111827" if theme == "light" else "#f9fafb"
        self.muted = "#5f6b7a" if theme == "light" else "#cbd5e1"
        self.border = "#d8dee8" if theme == "light" else "#374151"
        self.accent = "#2563eb"
        self.root.configure(bg=self.bg)

        self.video_var = tk.StringVar(value=str(video_path) if video_path is not None else "No video selected")
        self.model_var = tk.StringVar(value=model)
        self.device_var = tk.StringVar(value=device)
        self.auto_pause_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0)

        self._build_styles()
        self._build_layout()
        self._refresh_recent_outputs()
        self._update_run_state()
        self.root.after(150, self._poll_worker)

    def _build_styles(self) -> None:
        if hasattr(self.ttk, "Window"):
            style = self.root.style
        else:
            style = self.ttk.Style(self.root)
            try:
                style.theme_use("clam")
            except Exception:
                pass

        style.configure("Easy.TFrame", background=self.bg)
        style.configure("Panel.TLabelframe", background=self.bg, bordercolor=self.border)
        style.configure(
            "Panel.TLabelframe.Label", background=self.bg, foreground=self.text, font=("Segoe UI Semibold", 12)
        )
        style.configure("EasyTitle.TLabel", background=self.bg, foreground=self.text, font=("Segoe UI Semibold", 24))
        style.configure("EasyBody.TLabel", background=self.bg, foreground=self.muted, font=("Segoe UI", 10))
        style.configure("PanelBody.TLabel", background=self.surface, foreground=self.muted, font=("Segoe UI", 10))
        style.configure(
            "PanelTitle.TLabel", background=self.surface, foreground=self.text, font=("Segoe UI Semibold", 11)
        )
        style.configure("Status.TLabel", background=self.surface, foreground=self.text, font=("Segoe UI", 10))
        style.configure("Primary.TButton", font=("Segoe UI Semibold", 11), padding=(14, 10))
        style.configure("Plain.TButton", font=("Segoe UI", 10), padding=(10, 8))
        style.configure("Easy.TCheckbutton", background=self.surface, foreground=self.text, font=("Segoe UI", 10))

    def _build_layout(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = self.ttk.Frame(root, style="Easy.TFrame", padding=(24, 20, 24, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.ttk.Label(header, text="Snatch Technique Evaluator", style="EasyTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.ttk.Label(
            header,
            text="Side-view video analysis with pose tracking, bar path output, and technique scoring.",
            style="EasyBody.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        main = self.ttk.Frame(root, style="Easy.TFrame", padding=(24, 0, 24, 18))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        analysis_panel = self.ttk.Labelframe(main, text="Analysis", style="Panel.TLabelframe", padding=(18, 16))
        analysis_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        analysis_panel.columnconfigure(0, weight=1)
        analysis_panel.rowconfigure(5, weight=1)

        self.ttk.Label(analysis_panel, text="Video", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        video_row = self.ttk.Frame(analysis_panel, style="Easy.TFrame")
        video_row.grid(row=1, column=0, sticky="ew", pady=(8, 14))
        video_row.columnconfigure(0, weight=1)
        self.video_label = self.ttk.Label(
            video_row, textvariable=self.video_var, style="PanelBody.TLabel", wraplength=600
        )
        self.video_label.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.browse_button = self.ttk.Button(
            video_row, text="Browse Video", style="Plain.TButton", command=self._browse_video
        )
        self.browse_button.grid(row=0, column=1, sticky="e")

        settings = self.ttk.Frame(analysis_panel, style="Easy.TFrame")
        settings.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)
        self.ttk.Label(settings, text="Model", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.model_combo = self.ttk.Combobox(
            settings, textvariable=self.model_var, values=self.model_choices, state="readonly", width=22
        )
        self.model_combo.grid(row=0, column=1, sticky="ew", padx=(0, 18))
        self.ttk.Label(settings, text="Device", style="PanelTitle.TLabel").grid(
            row=0, column=2, sticky="w", padx=(0, 8)
        )
        self.device_combo = self.ttk.Combobox(
            settings, textvariable=self.device_var, values=self.device_choices, state="readonly", width=12
        )
        self.device_combo.grid(row=0, column=3, sticky="ew")

        options = self.ttk.Frame(analysis_panel, style="Easy.TFrame")
        options.grid(row=3, column=0, sticky="ew", pady=(0, 16))
        self.ttk.Checkbutton(
            options,
            text="Pause automatically after stable recovery",
            variable=self.auto_pause_var,
            style="Easy.TCheckbutton",
        ).grid(row=0, column=0, sticky="w")

        action_row = self.ttk.Frame(analysis_panel, style="Easy.TFrame")
        action_row.grid(row=4, column=0, sticky="ew", pady=(0, 18))
        action_row.columnconfigure(0, weight=1)
        self.run_button = self.ttk.Button(
            action_row, text="Track Bar Path", style="Primary.TButton", command=self._start_analysis
        )
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.open_last_button = self.ttk.Button(
            action_row, text="Open Output", style="Plain.TButton", command=self._open_last_output
        )
        self.open_last_button.grid(row=0, column=1, sticky="e")

        status_panel = self.ttk.Labelframe(
            analysis_panel, text="Run Status", style="Panel.TLabelframe", padding=(12, 10)
        )
        status_panel.grid(row=5, column=0, sticky="nsew")
        status_panel.columnconfigure(0, weight=1)
        status_panel.rowconfigure(2, weight=1)
        self.ttk.Label(status_panel, textvariable=self.status_var, style="Status.TLabel").grid(
            row=0, column=0, sticky="ew"
        )
        self.progress_bar = self.ttk.Progressbar(
            status_panel, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(8, 10))
        self.log_text = self.tk.Text(
            status_panel,
            height=10,
            bg="#101820",
            fg="#f6f7f9",
            insertbackground="#f6f7f9",
            relief="flat",
            font=("Consolas", 10),
            wrap="word",
        )
        self.log_text.grid(row=2, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")
        self._append_log("Ready.")

        outputs_panel = self.ttk.Labelframe(main, text="Outputs", style="Panel.TLabelframe", padding=(14, 12))
        outputs_panel.grid(row=0, column=1, sticky="nsew")
        outputs_panel.columnconfigure(0, weight=1)
        outputs_panel.rowconfigure(1, weight=1)
        self.ttk.Label(outputs_panel, text="Recent analyses", style="PanelTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )

        list_frame = self.ttk.Frame(outputs_panel, style="Easy.TFrame")
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 12))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.recent_listbox = self.tk.Listbox(
            list_frame,
            height=10,
            activestyle="dotbox",
            bg=self.surface,
            fg=self.text,
            selectbackground=self.accent,
            selectforeground="#ffffff",
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            font=("Segoe UI", 10),
        )
        self.recent_listbox.grid(row=0, column=0, sticky="nsew")
        self.recent_listbox.bind("<<ListboxSelect>>", lambda _event: self._update_output_buttons())
        scrollbar = self.ttk.Scrollbar(list_frame, orient="vertical", command=self.recent_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.recent_listbox.configure(yscrollcommand=scrollbar.set)

        output_actions = self.ttk.Frame(outputs_panel, style="Easy.TFrame")
        output_actions.grid(row=2, column=0, sticky="ew")
        output_actions.columnconfigure((0, 1), weight=1)
        self.open_selected_button = self.ttk.Button(
            output_actions, text="Open Selected", style="Plain.TButton", command=self._open_selected_output
        )
        self.open_selected_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.refresh_button = self.ttk.Button(
            output_actions, text="Refresh", style="Plain.TButton", command=self._refresh_recent_outputs
        )
        self.refresh_button.grid(row=0, column=1, sticky="ew")

        self.summary_text = self.tk.Text(
            outputs_panel,
            height=8,
            bg=self.surface,
            fg=self.text,
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 10),
            wrap="word",
        )
        self.summary_text.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        self.summary_text.insert("1.0", "No completed run in this session.")
        self.summary_text.configure(state="disabled")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _browse_video(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="Select a side-view snatch video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv"),
                ("MP4", "*.mp4"),
                ("MOV", "*.mov"),
                ("AVI", "*.avi"),
                ("MKV", "*.mkv"),
            ],
        )
        if not selected:
            return
        self.selected_video = Path(selected)
        self.video_var.set(str(self.selected_video))
        self._append_log(f"Selected video: {self.selected_video.name}")
        self._update_run_state()

    def _start_analysis(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.messagebox.showinfo("Analysis Running", "The current analysis is still running.")
            return
        if self.selected_video is None:
            self._browse_video()
        if self.selected_video is None:
            return
        if not self.selected_video.exists():
            self.messagebox.showerror("Video Not Found", f"Could not find:\n{self.selected_video}")
            return

        video_path = self.selected_video
        pose_model = self.model_var.get()
        device = self.device_var.get()
        auto_pause = self.auto_pause_var.get()

        self.progress_var.set(0)
        self.status_var.set("Running. Click the barbell center in the video window when prompted.")
        self._append_log(f"Starting analysis: {video_path.name}")
        self._append_log(f"Model: {pose_model} | Device: {device}")
        self._set_busy(True)

        def _run() -> None:
            try:
                AnalyzerConfig, SnatchAnalysisSession = _load_analysis_symbols()

                def progress_callback(update: dict[str, Any]) -> None:
                    self.events.put(("progress", update))

                session = SnatchAnalysisSession(
                    AnalyzerConfig(
                        video_path=video_path,
                        pose_backend_name="yolo",
                        model_name=_model_for_backend(pose_model),
                        device=_preferred_device() if device == "auto" else device,
                        output_dir=None,
                        reference_path=None,
                        auto_pause=auto_pause,
                        pause_mode="after-recovery-stable",
                        interactive=True,
                        show_live_window=True,
                        persist_outputs=True,
                        progress_callback=progress_callback,
                    )
                )
                summary = session.run()
                self.events.put(("success", {"output_dir": str(session.output_dir), "summary": summary}))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self.worker = threading.Thread(target=_run, daemon=True)
        self.worker.start()

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        readonly = "disabled" if busy else "readonly"
        self.browse_button.configure(state=state)
        self.run_button.configure(state=state if self.selected_video is not None else "disabled")
        self.model_combo.configure(state=readonly)
        self.device_combo.configure(state=readonly)

    def _update_run_state(self) -> None:
        busy = self.worker is not None and self.worker.is_alive()
        self.run_button.configure(state="normal" if self.selected_video is not None and not busy else "disabled")
        self.open_last_button.configure(state="normal" if self.last_output_dir is not None else "disabled")

    def _poll_worker(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "progress":
                    self._handle_progress(payload)
                elif event == "success":
                    self._handle_success(payload)
                else:
                    self._handle_error(str(payload))
        except queue.Empty:
            pass
        self.root.after(150, self._poll_worker)

    def _handle_progress(self, payload: dict[str, Any]) -> None:
        progress = payload.get("progress")
        if progress is not None:
            self.progress_var.set(float(progress))
        state = payload.get("live_state") or "processing"
        frame_index = payload.get("frame_index")
        frame_count = payload.get("frame_count")
        if isinstance(frame_index, int) and isinstance(frame_count, int) and frame_count > 0:
            self.status_var.set(f"Processing frame {frame_index + 1:,} of {frame_count:,} ({state}).")
        else:
            self.status_var.set(f"Processing video ({state}).")

    def _handle_success(self, payload: dict[str, Any]) -> None:
        self._set_busy(False)
        self.progress_var.set(100)
        self.last_output_dir = Path(payload["output_dir"])
        summary = payload["summary"]
        self.status_var.set("Analysis complete.")
        self._append_log(f"Completed: {self.last_output_dir}")
        self._render_summary(summary, self.last_output_dir)
        self._refresh_recent_outputs()
        self._update_run_state()

    def _handle_error(self, message: str) -> None:
        self._set_busy(False)
        self.status_var.set("Analysis failed.")
        self._append_log("Error: " + message)
        self._update_run_state()
        self.messagebox.showerror("Analysis Failed", message)

    def _render_summary(self, summary: dict[str, Any], output_dir: Path) -> None:
        score = summary.get("score")
        score_text = "--" if score is None else f"{float(score):.0f}/100"
        warnings = summary.get("warnings", [])
        completion = summary.get("completion_frame")
        lines = [
            f"Output: {output_dir}",
            f"Score: {score_text}",
            f"Warnings: {len(warnings)}",
            f"Completion frame: {completion if completion is not None else 'not detected'}",
            "",
            "Files:",
            "annotated.mp4",
            "frames.csv",
            "metrics.json",
        ]
        if warnings:
            lines.extend(["", "Top warnings:"])
            lines.extend(f"- {warning}" for warning in warnings[:3])

        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.configure(state="disabled")

    def _outputs_root(self) -> Path:
        return Path(__file__).resolve().parent.parent / "outputs"

    def _refresh_recent_outputs(self) -> None:
        outputs_root = self._outputs_root()
        if not outputs_root.exists():
            self.recent_outputs = []
        else:
            dirs = [path for path in outputs_root.iterdir() if path.is_dir()]
            self.recent_outputs = sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)[:8]

        self.recent_listbox.delete(0, "end")
        if not self.recent_outputs:
            self.recent_listbox.insert("end", "No analyses yet")
        else:
            for output_dir in self.recent_outputs:
                self.recent_listbox.insert("end", self._describe_output(output_dir))
        self._update_output_buttons()

    def _describe_output(self, output_dir: Path) -> str:
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            return output_dir.name
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return output_dir.name
        video_name = Path(str(data.get("video_path") or output_dir.name)).name
        score = data.get("score")
        score_text = "--" if score is None else f"{float(score):.0f}"
        return f"{score_text}/100  {video_name}"

    def _update_output_buttons(self) -> None:
        has_outputs = bool(self.recent_outputs)
        has_selection = bool(self.recent_listbox.curselection()) and has_outputs
        self.open_selected_button.configure(state="normal" if has_selection else "disabled")
        self.open_last_button.configure(
            state="normal" if self.last_output_dir is not None or has_outputs else "disabled"
        )

    def _selected_output_dir(self) -> Path | None:
        selection = self.recent_listbox.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if index >= len(self.recent_outputs):
            return None
        return self.recent_outputs[index]

    def _open_selected_output(self) -> None:
        output_dir = self._selected_output_dir()
        if output_dir is None:
            return
        self._open_path(output_dir)

    def _open_last_output(self) -> None:
        output_dir = self.last_output_dir
        if output_dir is None and self.recent_outputs:
            output_dir = self.recent_outputs[0]
        if output_dir is None:
            return
        self._open_path(output_dir)

    def _open_path(self, path: Path) -> None:
        if not path.exists():
            self.messagebox.showerror("Output Missing", f"Could not find:\n{path}")
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            self.messagebox.showerror("Open Failed", str(exc))

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        if args.mode in {"launch", "dashboard"}:
            launcher = EasySnatchLauncher(
                model=args.model,
                theme=args.theme,
                video_path=args.video,
                device=args.device,
            )
            return launcher.run()

        AnalyzerConfig, SnatchAnalysisSession = _load_analysis_symbols()

        video_path = args.video or pick_video()
        if video_path is None:
            print("No video selected.")
            return 1
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return 1

        session = SnatchAnalysisSession(
            AnalyzerConfig(
                video_path=video_path,
                pose_backend_name=args.backend,
                model_name=_model_for_backend(args.model),
                device=args.device,
                output_dir=args.output_dir,
                reference_path=None,
                auto_pause=True,
                pause_mode="after-recovery-stable",
                interactive=True,
                show_live_window=True,
                persist_outputs=True,
            )
        )
        summary = session.run()
    except Exception as exc:
        print(f"Snatch detector failed: {exc}", file=sys.stderr)
        return 1

    print(f"Tracking complete. Outputs written to: {session.output_dir}")
    print(f"Warnings: {len(summary.get('warnings', []))}")
    if summary.get("completion_frame") is not None:
        print(f"Completion frame: {summary['completion_frame']}")
    else:
        print("Completion frame: not detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
