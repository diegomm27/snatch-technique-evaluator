from __future__ import annotations

"""Analysis Configuration screen per WEIA-17 wireframes.

Allows users to select preset profiles, adjust analysis parameters,
and configure output preferences before running analysis.
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
    COLOR_BG_LIGHT,
    COLOR_BG_DARK,
    COLOR_SURFACE_LIGHT,
    COLOR_SURFACE_DARK,
    COLOR_TEXT_PRIMARY_LIGHT,
    COLOR_TEXT_PRIMARY_DARK,
    COLOR_TEXT_SECONDARY_LIGHT,
    COLOR_TEXT_SECONDARY_DARK,
)


class AnalysisPreset:
    """Represents an analysis preset profile."""

    def __init__(self, name: str, description: str, params: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.params = params


class AnalysisConfiguration:
    """Analysis Configuration screen per WEIA-17 wireframes.

    Layout:
    - Left panel: Preset profiles (Beginner, Intermediate, Advanced, Olympic)
    - Right panel: Adjustable parameters (collapsible advanced section)
    - Bottom: Save as Preset, Run Analysis buttons
    """

    PRESETS = {
        "Beginner": AnalysisPreset(
            name="Beginner",
            description="Standard settings optimized for beginner lifters. "
                        "More lenient thresholds for phase detection.",
            params={
                "pose_model": "yolo11n-pose.pt",
                "tracking_sensitivity": 0.5,
                "auto_pause": True,
                "pause_mode": "after-recovery-stable",
                "output_format": "pdf",
            },
        ),
        "Intermediate": AnalysisPreset(
            name="Intermediate",
            description="Balanced settings for intermediate lifters with moderate thresholds.",
            params={
                "pose_model": "yolo11n-pose.pt",
                "tracking_sensitivity": 0.7,
                "auto_pause": True,
                "pause_mode": "after-recovery-stable",
                "output_format": "pdf",
            },
        ),
        "Advanced": AnalysisPreset(
            name="Advanced",
            description="Tight thresholds for advanced lifters. Requires clean video quality.",
            params={
                "pose_model": "yolo11x-pose.pt",
                "tracking_sensitivity": 0.9,
                "auto_pause": True,
                "pause_mode": "after-recovery-stable",
                "output_format": "pdf",
            },
        ),
        "Olympic": AnalysisPreset(
            name="Olympic",
            description="Maximum precision settings. Uses the largest model for best accuracy.",
            params={
                "pose_model": "yolo11x-pose.pt",
                "tracking_sensitivity": 1.0,
                "auto_pause": True,
                "pause_mode": "after-recovery-stable",
                "output_format": "pdf",
            },
        ),
    }

    def __init__(
        self,
        video_path: str = "",
        on_run_analysis=None,
        on_back=None,
        theme: str = "light",
    ) -> None:
        self.video_path = video_path
        self.on_run_analysis = on_run_analysis
        self.on_back = on_back
        self.theme_name = theme
        self.selected_preset = "Intermediate"

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

        self.root.title("Analysis Configuration - Weight Lifting Analyzer")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

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

        style.configure("Config.TFrame", background=theme["bg"])

        # Preset cards
        style.configure(
            "PresetCard.TFrame",
            background=theme["surface"],
        )
        style.configure(
            "PresetCardSelected.TFrame",
            background=f"{COLOR_PRIMARY_600}18",
        )
        style.configure(
            "PresetTitle.TLabel",
            background=theme["surface"],
            foreground=theme["text_primary"],
            font=("Segoe UI Semibold", 14),
        )
        style.configure(
            "PresetDesc.TLabel",
            background=theme["surface"],
            foreground=theme["text_secondary"],
            font=("Segoe UI", 10),
        )

        # Parameters
        style.configure(
            "ParamSection.TFrame",
            background=theme["surface"],
        )
        style.configure(
            "ParamLabel.TLabel",
            background=theme["surface"],
            foreground=theme["text_primary"],
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "ParamValue.TLabel",
            background=theme["surface"],
            foreground=theme["text_secondary"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "ParamSectionHeader.TLabel",
            background=theme["surface"],
            foreground=theme["text_primary"],
            font=("Segoe UI Semibold", 13),
        )

        # Buttons
        style.configure(
            "ConfigPrimary.TButton",
            font=("Segoe UI Semibold", 13),
            padding=(20, 10),
        )
        style.configure(
            "ConfigSecondary.TButton",
            font=("Segoe UI", 11),
            padding=(14, 8),
        )
        style.configure(
            "ConfigGhost.TButton",
            font=("Segoe UI", 11),
            padding=(10, 6),
        )

    def _build_layout(self, theme: dict) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        # ── Top Bar ────────────────────────────────────────────────────
        top_bar = self.ttk.Frame(root, style="Config.TFrame", padding=(20, 14, 20, 10))
        top_bar.grid(row=0, column=0, sticky="ew")
        top_bar.columnconfigure(1, weight=1)

        self.ttk.Label(
            top_bar,
            text="Analysis Configuration",
            style="ParamSectionHeader.TLabel",
        ).grid(row=0, column=0, sticky="w")

        back_btn = self.ttk.Button(
            top_bar,
            text="\u2190 Back",
            style="ConfigGhost.TButton",
            command=self._on_back,
        )
        back_btn.grid(row=0, column=2, sticky="e")

        # ── Content ────────────────────────────────────────────────────
        content = self.ttk.Frame(root, style="Config.TFrame", padding=(20, 12, 20, 16))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=3)
        content.rowconfigure(0, weight=1)

        # ── Left Panel: Preset Profiles ────────────────────────────────
        presets_panel = self.ttk.Frame(content, style="Config.TFrame", padding=(0, 0, 12, 0))
        presets_panel.grid(row=0, column=0, sticky="nsew")
        presets_panel.columnconfigure(0, weight=1)
        presets_panel.rowconfigure(0, weight=0)
        presets_panel.rowconfigure(1, weight=1)

        self.ttk.Label(
            presets_panel,
            text="Preset Profiles",
            style="ParamSectionHeader.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        presets_frame = self.ttk.Frame(presets_panel, style="Config.TFrame")
        presets_frame.grid(row=1, column=0, sticky="nsew")
        presets_frame.columnconfigure(0, weight=1)
        presets_frame.rowconfigure(0, weight=1)

        # Scrollable presets area
        try:
            from tkinter import scrolledtext as st
            has_scrolled = True
        except ImportError:
            has_scrolled = False

        self._preset_cards: dict[str, Any] = {}
        for idx, (name, preset) in enumerate(self.PRESETS.items()):
            card = self._create_preset_card(presets_frame, name, preset, idx)
            card.grid(row=0, column=0, sticky="nsew", pady=(0 if idx == 0 else 8, 0))

        # ── Right Panel: Parameters ────────────────────────────────────
        params_panel = self.ttk.Frame(content, style="Config.TFrame", padding=(12, 0, 0, 0))
        params_panel.grid(row=0, column=1, sticky="nsew")
        params_panel.columnconfigure(0, weight=1)
        params_panel.rowconfigure(0, weight=0)
        params_panel.rowconfigure(1, weight=1)

        self.ttk.Label(
            params_panel,
            text="Parameters",
            style="ParamSectionHeader.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        params_frame = self.ttk.Frame(params_panel, style="Config.TFrame")
        params_frame.grid(row=1, column=0, sticky="nsew")
        params_frame.columnconfigure(0, weight=1)
        params_frame.rowconfigure(0, weight=1)

        self._build_parameters(params_frame, theme)

        # ── Bottom Bar ─────────────────────────────────────────────────
        bottom_bar = self.ttk.Frame(root, style="Config.TFrame", padding=(20, 10, 20, 16))
        bottom_bar.grid(row=2, column=0, sticky="ew")
        bottom_bar.columnconfigure(1, weight=1)

        self.ttk.Button(
            bottom_bar,
            text="Save as Preset",
            style="ConfigSecondary.TButton",
            command=self._save_preset,
        ).grid(row=0, column=0, padx=(0, 12))

        self.ttk.Button(
            bottom_bar,
            text="Run Analysis",
            style="ConfigPrimary.TButton",
            command=self._run_analysis,
        ).grid(row=0, column=2, sticky="e")

    def _create_preset_card(
        self, parent: Any, name: str, preset: AnalysisPreset, index: int
    ) -> Any:
        """Create a single preset profile card."""
        is_selected = name == self.selected_preset
        card = self.ttk.Frame(
            parent,
            style="PresetCardSelected.TFrame" if is_selected else "PresetCard.TFrame",
            padding=(16, 14),
        )
        card.columnconfigure(0, weight=1)

        # Radio button selection
        radio = self.tk.Radiobutton(
            card,
            variable=self.tk.StringVar(value=self.selected_preset),
            value=name,
            command=lambda n=name: self._select_preset(n),
            bg=card.cget("bg") if hasattr(card, 'cget') else "#ffffff",
            fg=COLOR_PRIMARY_600 if is_selected else COLOR_TEXT_PRIMARY_LIGHT,
            font=("Segoe UI Semibold", 13),
            selectcolor=card.cget("bg") if hasattr(card, 'cget') else "#eee",
        )
        # For ttkb, we need a different approach
        if self._has_ttkb:
            radio = self.ttk.Radiobutton(
                card,
                text=name,
                variable=self.tk.StringVar(value=self.selected_preset),
                value=name,
                command=lambda n=name: self._select_preset(n),
                style="PresetRadio.TButton" if is_selected else "",
            )
            # Use a label instead for better visual
            label = self.ttk.Label(
                card,
                text=name,
                style="PresetTitle.TLabel",
            )
            label.grid(row=0, column=0, sticky="w")
            card.bind("<Button-1>", lambda e, n=name: self._select_preset(n))
        else:
            radio.grid(row=0, column=0, sticky="w")

        self.ttk.Label(
            card,
            text=preset.description,
            style="PresetDesc.TLabel",
            wraplength=340,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        self._preset_cards[name] = card
        return card

    def _select_preset(self, name: str) -> None:
        """Handle preset selection."""
        self.selected_preset = name
        for n, card in self._preset_cards.items():
            theme_dict = THEME_LIGHT if self.theme_name == "light" else THEME_DARK
            bg = f"{COLOR_PRIMARY_600}18" if n == name else theme_dict["surface"]
            card.configure(background=bg)

        # Update parameter values from preset
        preset = self.PRESETS[name]
        for key, value in preset.params.items():
            var_name = f"_{key}_var"
            if hasattr(self, var_name):
                var = getattr(self, var_name)
                if isinstance(var, self.tk.StringVar):
                    var.set(str(value))
                elif isinstance(var, self.tk.DoubleVar):
                    var.set(float(value))

    def _build_parameters(self, parent: Any, theme: dict) -> None:
        """Build the adjustable parameters section."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=0)
        parent.rowconfigure(1, weight=0)
        parent.rowconfigure(2, weight=0)
        parent.rowconfigure(3, weight=1)

        # ── Basic Parameters ───────────────────────────────────────────
        basic_frame = self.ttk.Frame(parent, style="ParamSection.TFrame", padding=(16, 14))
        basic_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        basic_frame.columnconfigure(1, weight=1)

        self.ttk.Label(
            basic_frame,
            text="Basic Settings",
            style="ParamSectionHeader.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Pose model
        self._pose_model_var = self.tk.StringVar(value="yolo11n-pose.pt")
        row = 1
        self.ttk.Label(
            basic_frame, text="Pose Model", style="ParamLabel.TLabel"
        ).grid(row=row, column=0, sticky="w", pady=4)
        model_cb = self.ttk.Combobox(
            basic_frame,
            textvariable=self._pose_model_var,
            values=["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11x-pose.pt"],
            state="readonly",
            font=("Segoe UI", 10),
        )
        model_cb.grid(row=row, column=1, sticky="ew", padx=(12, 0))
        self.ttk.Label(
            basic_frame,
            text="n=fast / s=balanced / x=best accuracy",
            font=("Segoe UI", 8),
            foreground=COLOR_TEXT_SECONDARY_LIGHT,
        ).grid(row=row + 1, column=0, columnspan=2, sticky="w", pady=(2, 0))

        # Tracking sensitivity
        row = 3
        self.ttk.Label(
            basic_frame, text="Tracking Sensitivity", style="ParamLabel.TLabel"
        ).grid(row=row, column=0, sticky="w", pady=4)
        self._sensitivity_var = self.tk.DoubleVar(value=0.7)
        sensitivity_scale = self.ttk.Scale(
            basic_frame,
            from_=0.0,
            to=1.0,
            variable=self._sensitivity_var,
            orient="horizontal",
        )
        sensitivity_scale.grid(row=row, column=1, sticky="ew", padx=(12, 0))
        self._sensitivity_label = self.ttk.Label(
            basic_frame, text="0.7", style="ParamValue.TLabel"
        )
        self._sensitivity_label.grid(row=row, column=2, sticky="w", padx=(8, 0))
        sensitivity_scale.configure(command=self._update_sensitivity_label)

        # Auto pause
        row = 4
        self._auto_pause_var = self.tk.BooleanVar(value=True)
        self.ttk.Label(
            basic_frame, text="Auto-pause after lift", style="ParamLabel.TLabel"
        ).grid(row=row, column=0, sticky="w", pady=4)
        pause_check = self.ttk.Checkbutton(
            basic_frame,
            variable=self._auto_pause_var,
            text="Enabled",
            style="Switch.TCheckbutton",
        )
        pause_check.grid(row=row, column=1, sticky="w", padx=(12, 0))

        # ── Advanced Parameters ────────────────────────────────────────
        self._advanced_frame = self.ttk.Frame(parent, style="ParamSection.TFrame", padding=(16, 14))
        self._advanced_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self._advanced_frame.columnconfigure(0, weight=1)

        self.ttk.Label(
            self._advanced_frame,
            text="Advanced Settings",
            style="ParamSectionHeader.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        self._advanced_frame.columnconfigure(1, weight=1)

        self._reference_var = self.tk.StringVar(value="default_reference.json")
        row = 1
        self.ttk.Label(
            self._advanced_frame, text="Reference Profile", style="ParamLabel.TLabel"
        ).grid(row=row, column=0, sticky="w", pady=4)
        ref_cb = self.ttk.Combobox(
            self._advanced_frame,
            textvariable=self._reference_var,
            values=["default_reference.json", "custom_reference.json"],
            state="readonly",
            font=("Segoe UI", 10),
        )
        ref_cb.grid(row=row, column=1, sticky="ew", padx=(12, 0))

        row = 2
        self._output_format_var = self.tk.StringVar(value="pdf")
        self.ttk.Label(
            self._advanced_frame, text="Output Format", style="ParamLabel.TLabel"
        ).grid(row=row, column=0, sticky="w", pady=4)
        output_cb = self.ttk.Combobox(
            self._advanced_frame,
            textvariable=self._output_format_var,
            values=["pdf", "csv", "json"],
            state="readonly",
            font=("Segoe UI", 10),
        )
        output_cb.grid(row=row, column=1, sticky="ew", padx=(12, 0))

        row = 3
        self._device_var = self.tk.StringVar(value="auto")
        self.ttk.Label(
            self._advanced_frame, text="Device", style="ParamLabel.TLabel"
        ).grid(row=row, column=0, sticky="w", pady=4)
        device_cb = self.ttk.Combobox(
            self._advanced_frame,
            textvariable=self._device_var,
            values=["auto", "cpu", "cuda", "cuda:0"],
            state="readonly",
            font=("Segoe UI", 10),
        )
        device_cb.grid(row=row, column=1, sticky="ew", padx=(12, 0))

        # Hide advanced by default
        self._advanced_visible = False
        self._advanced_frame.grid_remove()

        # Toggle button
        self.ttk.Button(
            parent,
            text="Show Advanced \u25BC",
            style="ConfigGhost.TButton",
            command=self._toggle_advanced,
        ).grid(row=2, column=0, sticky="w", pady=(0, 8))

        # ── Video Path Display ─────────────────────────────────────────
        video_frame = self.ttk.Frame(parent, style="ParamSection.TFrame", padding=(16, 14))
        video_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        video_frame.columnconfigure(0, weight=1)

        self.ttk.Label(
            video_frame,
            text="Source Video",
            style="ParamSectionHeader.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self._video_path_label = self.ttk.Label(
            video_frame,
            text=self.video_path or "No video selected",
            style="ParamValue.TLabel",
            wraplength=500,
        )
        self._video_path_label.grid(row=1, column=0, sticky="w")

    def _update_sensitivity_label(self, value: str) -> None:
        self._sensitivity_label.config(text=f"{float(value):.1f}")

    def _toggle_advanced(self) -> None:
        self._advanced_visible = not self._advanced_visible
        if self._advanced_visible:
            self._advanced_frame.grid()
            self._advanced_frame.tk.eval("tk::PlaceWindow %s center" % self._advanced_frame.winfo_pathname(0))
            self.ttk.Button(
                self._advanced_frame.master,
                text="Hide Advanced \u25B2",
                style="ConfigGhost.TButton",
                command=self._toggle_advanced,
            ).grid_forget()
        else:
            self._advanced_frame.grid_remove()

    def _save_preset(self) -> None:
        self.messagebox.showinfo(
            "Save Preset",
            f"Preset '{self.selected_preset}' configuration saved.",
        )

    def _run_analysis(self) -> None:
        if self.on_run_analysis:
            self.on_run_analysis(
                preset=self.selected_preset,
                pose_model=self._pose_model_var.get(),
                sensitivity=self._sensitivity_var.get(),
                auto_pause=self._auto_pause_var.get(),
                reference=self._reference_var.get(),
                output_format=self._output_format_var.get(),
                device=self._device_var.get(),
            )
        else:
            self.messagebox.showinfo(
                "Run Analysis",
                f"Running analysis with preset: {self.selected_preset}",
            )

    def run(self) -> int:
        self.root.mainloop()
        return 0
