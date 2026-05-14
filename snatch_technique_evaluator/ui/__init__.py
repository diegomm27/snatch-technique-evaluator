from __future__ import annotations

"""UI module for the Weight Lifting Analyzer.

Contains all screens per WEIA-17 UI/UX Design Plan:
- design_system: Color palette, typography, spacing, component styles
- home_screen: Home / Dashboard with recent analyses and quick stats
- results_dashboard: 3-zone results layout (video + metrics + actions)
- analysis_config: Analysis configuration with presets and parameters
- comparison_view: Side-by-side comparison mode
"""

from .design_system import (
    THEME_LIGHT,
    THEME_DARK,
    get_theme,
    COLOR_PRIMARY_600,
    COLOR_PRIMARY_500,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_ERROR,
    COLOR_INFO,
    TYPE_SCALE,
    SPACING,
    BUTTON_STYLES,
    CARD_STYLES,
    INPUT_STYLES,
    SCORE_COLORS,
    FONT_FAMILY,
    FONT_FAMILY_FALLBACK,
    WINDOW_MIN_WIDTH,
    WINDOW_MIN_HEIGHT,
    WINDOW_RECOMMENDED_WIDTH,
    WINDOW_RECOMMENDED_HEIGHT,
)

from .home_screen import HomeScreen, RecentAnalysis

from .results_dashboard import ResultsDashboard, PhaseMetric, KeyMetric

from .analysis_config import AnalysisConfiguration, AnalysisPreset

from .comparison_view import ComparisonView, ComparisonEntry

__all__ = [
    # Design system
    "THEME_LIGHT",
    "THEME_DARK",
    "get_theme",
    "COLOR_PRIMARY_600",
    "COLOR_PRIMARY_500",
    "COLOR_SUCCESS",
    "COLOR_WARNING",
    "COLOR_ERROR",
    "COLOR_INFO",
    "TYPE_SCALE",
    "SPACING",
    "BUTTON_STYLES",
    "CARD_STYLES",
    "INPUT_STYLES",
    "SCORE_COLORS",
    "FONT_FAMILY",
    "FONT_FAMILY_FALLBACK",
    "WINDOW_MIN_WIDTH",
    "WINDOW_MIN_HEIGHT",
    "WINDOW_RECOMMENDED_WIDTH",
    "WINDOW_RECOMMENDED_HEIGHT",
    # Screens
    "HomeScreen",
    "RecentAnalysis",
    "ResultsDashboard",
    "PhaseMetric",
    "KeyMetric",
    "AnalysisConfiguration",
    "AnalysisPreset",
    "ComparisonView",
    "ComparisonEntry",
]
