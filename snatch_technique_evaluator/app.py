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

    for target in (parser, analyze_parser):
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
    def __init__(self, model: str) -> None:
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
        self.worker: threading.Thread | None = None
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()

        if hasattr(ttk, "Window"):
            self.root = ttk.Window(themename="flatly")
        else:
            self.root = tk.Tk()
        self.root.title("Snatch Bar Path")
        self.root.geometry("900x560")
        self.root.minsize(780, 520)
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
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10), padding=(10, 8))
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

        header = self.ttk.Frame(root, style="App.TFrame", padding=(26, 22, 26, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.ttk.Label(header, text="Snatch Bar Path", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.ttk.Label(
            header,
            text=(
                "Choose a side-view lift video, click the barbell once, and generate the "
                "tracked bar path with annotated outputs."
            ),
            style="Body.TLabel",
            wraplength=760,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        content = self.ttk.Frame(root, style="App.TFrame", padding=(26, 8, 26, 18))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)

        analyze_card = self.ttk.Frame(content, style="Card.TFrame", padding=18)
        analyze_card.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=(0, 0))
        analyze_card.columnconfigure(0, weight=1)
        analyze_card.columnconfigure(1, weight=0)

        self.ttk.Label(analyze_card, text="Track Lift", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.ttk.Label(
            analyze_card,
            text=(
                "Pick a side-view snatch video. The app will ask for one initial barbell "
                "click, then track the bar and generate the path overlay."
            ),
            style="CardBody.TLabel",
            wraplength=470,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 16))

        self.video_var = self.tk.StringVar()
        video_entry = self.ttk.Entry(analyze_card, textvariable=self.video_var, font=("Segoe UI", 10))
        video_entry.grid(row=2, column=0, sticky="ew", padx=(0, 10))
        self.ttk.Button(
            analyze_card,
            text="Browse Video",
            style="Secondary.TButton",
            command=self._browse_video,
        ).grid(row=2, column=1, sticky="ew")

        self.model_var = self.tk.StringVar(value=self.model_default)

        model_row = self.ttk.Frame(analyze_card, style="Card.TFrame")
        model_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        model_row.columnconfigure(1, weight=1)
        self.ttk.Label(model_row, text="Model", style="CardBody.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.ttk.Combobox(
            model_row,
            textvariable=self.model_var,
            values=["yolo11n-pose.pt", "yolo11x-pose.pt"],
            state="readonly",
            font=("Segoe UI", 10),
        ).grid(row=0, column=1, sticky="ew")
        self.ttk.Label(
            model_row,
            text="yolo11n-pose.pt = fast / lower accuracy   |   yolo11x-pose.pt = slower / best accuracy",
            font=("Segoe UI", 8),
            foreground="#8899aa",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.ttk.Button(
            analyze_card,
            text="Track Bar Path",
            style="Accent.TButton",
            command=self._start_analyze,
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(18, 0))

        self.log_text = self.tk.Text(
            content,
            height=12,
            bg="#211915",
            fg="#f7efe2",
            insertbackground="#f7efe2",
            relief="flat",
            font=("Consolas", 10),
            wrap="word",
        )
        self.log_text.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        self.log_text.insert("1.0", "Launcher ready.\n")
        self.log_text.configure(state="disabled")

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
        if selected:
            self.video_var.set(selected)

    def _start_analyze(self) -> None:
        video_text = self.video_var.get().strip()
        if not video_text:
            self.messagebox.showerror("Missing Video", "Select a snatch attempt video first.")
            return
        video_path = Path(video_text)
        if not video_path.exists():
            self.messagebox.showerror("Video Not Found", f"Could not find:\n{video_path}")
            return

        self._run_task(
            title="Tracking bar path...",
            task=self._analyze_task,
            kwargs={
                "video_path": video_path,
                "auto_pause": True,
                "pause_mode": "after-recovery-stable",
                "model": _model_for_backend(self.model_var.get()),
                "device": _preferred_device(),
            },
        )

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

    def _analyze_task(
        self,
        video_path: Path,
        auto_pause: bool,
        pause_mode: str,
        model: str,
        device: str,
    ) -> dict[str, Any]:
        AnalyzerConfig, SnatchAnalysisSession = _load_analysis_symbols()
        session = SnatchAnalysisSession(
            AnalyzerConfig(
                video_path=video_path,
                pose_backend_name="yolo",
                model_name=model,
                device=device,
                reference_path=None,
                auto_pause=auto_pause,
                pause_mode=pause_mode,
                interactive=True,
                show_live_window=True,
                persist_outputs=True,
            )
        )
        summary = session.run()
        return {
            "kind": "analysis",
            "output_dir": str(session.output_dir),
            "summary": summary,
        }

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
