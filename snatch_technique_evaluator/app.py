from __future__ import annotations

import argparse
import queue
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
    subparsers = parser.add_subparsers(dest="mode")

    analyze_parser = subparsers.add_parser("analyze", help="Track the bar path in a snatch attempt.")
    launch_parser = subparsers.add_parser("launch", help="Open the desktop launcher UI.")
    dashboard_parser = subparsers.add_parser("dashboard", help="Open the home dashboard UI.")

    for target in (parser, analyze_parser, launch_parser, dashboard_parser):
        _add_common_analysis_args(target)

    launch_parser.add_argument(
        "--backend",
        default="yolo",
        choices=["yolo"],
        help="Pose backend to use.",
    )
    launch_parser.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="Pose model name or path.",
    )
    launch_parser.add_argument(
        "--device",
        default=_preferred_device(),
        help="Torch device, for example auto, cpu, cuda, or cuda:0.",
    )
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
        self.ttk.Label(header, text="Weight Lifting Analyzer", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
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
                self.events.put(("success", {"kind": "analysis", "output_dir": str(session.output_dir), "summary": summary}))
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        AnalyzerConfig, SnatchAnalysisSession = _load_analysis_symbols()

        if args.mode == "launch":
            launcher = SnatchLauncher(
                model=args.model,
                theme=args.theme,
            )
            return launcher.run()

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
