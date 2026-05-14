from __future__ import annotations

"""Design system tokens for the Weight Lifting Analyzer.

Colors, typography, spacing, and component styles per WEIA-17 UI/UX Design Plan.
"""

# ── Color Palette ──────────────────────────────────────────────────────

# Primary
COLOR_PRIMARY_600 = "#4F46E5"  # primary actions, active states
COLOR_PRIMARY_500 = "#6366F1"  # hover states, accents

# Semantic (colorblind-safe)
COLOR_SUCCESS = "#0D9488"      # success / good form
COLOR_WARNING = "#D97706"      # warning / needs work
COLOR_ERROR = "#E11D48"        # error / critical
COLOR_INFO = "#0284C7"         # info

# Neutral - Light mode
COLOR_BG_LIGHT = "#F8FAFC"     # slate-50
COLOR_SURFACE_LIGHT = "#FFFFFF"
COLOR_TEXT_PRIMARY_LIGHT = "#0F172A"  # slate-900
COLOR_TEXT_SECONDARY_LIGHT = "#64748B"  # slate-500
COLOR_BORDER_LIGHT = "#E2E8F0"     # slate-200

# Neutral - Dark mode
COLOR_BG_DARK = "#0F172A"       # slate-900
COLOR_SURFACE_DARK = "#1E293B"  # slate-800
COLOR_TEXT_PRIMARY_DARK = "#F1F5F9"  # slate-100
COLOR_TEXT_SECONDARY_DARK = "#94A3B8"  # slate-400
COLOR_BORDER_DARK = "#334155"       # slate-700

# ── Themes ─────────────────────────────────────────────────────────────

THEME_LIGHT = {
    "bg": COLOR_BG_LIGHT,
    "surface": COLOR_SURFACE_LIGHT,
    "text_primary": COLOR_TEXT_PRIMARY_LIGHT,
    "text_secondary": COLOR_TEXT_SECONDARY_LIGHT,
    "border": COLOR_BORDER_LIGHT,
    "primary": COLOR_PRIMARY_600,
    "primary_hover": COLOR_PRIMARY_500,
    "success": COLOR_SUCCESS,
    "warning": COLOR_WARNING,
    "error": COLOR_ERROR,
    "info": COLOR_INFO,
}

THEME_DARK = {
    "bg": COLOR_BG_DARK,
    "surface": COLOR_SURFACE_DARK,
    "text_primary": COLOR_TEXT_PRIMARY_DARK,
    "text_secondary": COLOR_TEXT_SECONDARY_DARK,
    "border": COLOR_BORDER_DARK,
    "primary": COLOR_PRIMARY_500,
    "primary_hover": COLOR_PRIMARY_600,
    "success": COLOR_SUCCESS,
    "warning": COLOR_WARNING,
    "error": COLOR_ERROR,
    "info": COLOR_INFO,
}

# ── Typography ─────────────────────────────────────────────────────────

FONT_FAMILY = "Inter"
FONT_FAMILY_FALLBACK = ("Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif")

TYPE_SCALE = {
    "h1": {"size": 28, "weight": "bold", "line_height": 1.2},
    "h2": {"size": 22, "weight": "bold", "line_height": 1.3},
    "h3": {"size": 18, "weight": "semibold", "line_height": 1.3},
    "body": {"size": 14, "weight": "regular", "line_height": 1.5},
    "caption": {"size": 12, "weight": "regular", "line_height": 1.4},
    "mono": {"size": 13, "weight": "regular", "line_height": 1.5},
}

# ── Spacing System (4px base unit) ─────────────────────────────────────

SPACING = {"xs": 4, "sm": 8, "md": 12, "base": 16, "lg": 24, "xl": 32, "2xl": 48, "3xl": 64}

# ── Component Styles ───────────────────────────────────────────────────

BUTTON_STYLES = {
    "primary": {
        "bg": COLOR_PRIMARY_600, "text": "#FFFFFF", "border": "none",
        "hover_bg": COLOR_PRIMARY_500, "font_weight": "semibold",
        "font_size": 14, "padding": (10, 16), "border_radius": 6,
    },
    "secondary": {
        "bg": "transparent", "text": COLOR_PRIMARY_600,
        "border": f"1px solid {COLOR_PRIMARY_600}",
        "hover_bg": f"{COLOR_PRIMARY_600}14", "font_weight": "regular",
        "font_size": 14, "padding": (8, 16), "border_radius": 6,
    },
    "ghost": {
        "bg": "transparent", "text": COLOR_TEXT_PRIMARY_LIGHT,
        "border": "none", "hover_bg": f"{COLOR_TEXT_PRIMARY_LIGHT}10",
        "font_weight": "regular", "font_size": 14, "padding": (8, 12),
        "border_radius": 4,
    },
    "danger": {
        "bg": COLOR_ERROR, "text": "#FFFFFF", "border": "none",
        "hover_bg": "#BE123C", "font_weight": "semibold",
        "font_size": 14, "padding": (10, 16), "border_radius": 6,
    },
}

CARD_STYLES = {
    "bg": COLOR_SURFACE_LIGHT,
    "border": f"1px solid {COLOR_BORDER_LIGHT}",
    "border_radius": 12, "padding": 20,
    "shadow": "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06)",
    "hover_shadow": "0 4px 12px rgba(0,0,0,0.1)",
}

INPUT_STYLES = {
    "bg": COLOR_SURFACE_LIGHT,
    "border": f"1px solid {COLOR_BORDER_LIGHT}",
    "border_radius": 6, "padding": (8, 12), "font_size": 14,
    "focus_border": COLOR_PRIMARY_500, "focus_ring": f"{COLOR_PRIMARY_500}40",
    "placeholder": COLOR_TEXT_SECONDARY_LIGHT,
}

SCORE_COLORS = {
    "excellent": COLOR_SUCCESS,    # >= 85
    "good": COLOR_SUCCESS,         # >= 80
    "average": COLOR_WARNING,      # >= 65
    "below_average": COLOR_WARNING,# >= 50
    "poor": COLOR_ERROR,           # < 50
}

# ── Constants ──────────────────────────────────────────────────────────

MIN_TOUCH_TARGET = 44  # px - accessibility
WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 800
WINDOW_RECOMMENDED_WIDTH = 1400
WINDOW_RECOMMENDED_HEIGHT = 900
ANIMATION_DURATION_FAST = 150
ANIMATION_DURATION_NORMAL = 200
ANIMATION_DURATION_SLOW = 300


def get_theme(theme_name: str = "light") -> dict:
    """Return the theme dict for the given theme name."""
    if theme_name == "dark":
        return THEME_DARK
    return THEME_LIGHT
