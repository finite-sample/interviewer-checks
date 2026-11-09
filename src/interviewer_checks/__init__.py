"""Find data problems tied to interviewers in surveys and survey experiments."""

from importlib.metadata import version

from interviewer_checks.clock import clock_flags, interview_clock
from interviewer_checks.experiments import assignment_balance, cluster_influence, effect_heterogeneity
from interviewer_checks.responses import interviewer_similarity, pattern_repetition, percent_match, straightlining
from interviewer_checks.screen import Screen, screen

__version__ = version("interviewer-checks")

__all__ = [
    "Screen",
    "__version__",
    "assignment_balance",
    "clock_flags",
    "cluster_influence",
    "effect_heterogeneity",
    "interview_clock",
    "interviewer_similarity",
    "pattern_repetition",
    "percent_match",
    "screen",
    "straightlining",
]
