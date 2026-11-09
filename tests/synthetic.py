"""Simulated face-to-face surveys with optional planted interviewer problems."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

BATTERY = ["p1", "p2", "p3", "p4"]


def _ordinal(latent: np.ndarray, categories: int) -> np.ndarray:
    cuts = np.linspace(-1.5, 1.5, categories - 1)
    return np.searchsorted(cuts, latent).astype(float)


def make_survey(
    *,
    areas: int = 30,
    interviewers_per_area: int = 2,
    interviews_per_interviewer: int = 50,
    questionnaire_items: int = 60,
    categories: int = 5,
    area_sd: float = 0.5,
    loading: float = 0.6,
    effect: float = -0.15,
    battery_level: float = 0.8,
    fabricators: Sequence[int] = (),
    fabricated_share: float = 0.6,
    fabricated_control: float = 0.85,
    copiers: Sequence[int] = (),
    copied_share: float = 0.4,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per interview: interviewer, area, treated, questionnaire items q*, battery p1-p4 (0-4), support.

    Answers come from an ordered-probit-like rule: latent = area shift (per item) + loading * respondent trait +
    noise, cut into ``categories`` bins, so items correlate through the trait and neighbours resemble each other
    through the area. Treatment shifts the battery's latent values by ``effect``; a higher ``battery_level``
    piles genuine answers up at the top of the scale, as in the Pakistan data.

    ``fabricators`` record, in ``fabricated_share`` of their interviews, the battery string 4-0-0-4 when the
    interview is control and 4-1-0-4 when treated, and ``fabricated_control`` of those invented
    interviews are control (the interviewer skips the randomization).
    ``copiers`` fill ``copied_share`` of their interviews by copying one of their own earlier interviews and
    changing about 5% of the questionnaire answers.
    """
    rng = np.random.default_rng(seed)
    interviewers = areas * interviewers_per_area
    n = interviewers * interviews_per_interviewer
    interviewer = np.repeat(np.arange(interviewers), interviews_per_interviewer)
    area = interviewer // interviewers_per_area
    treated = rng.random(n) < 0.5
    trait = rng.normal(size=n)

    area_shift = rng.normal(0, area_sd, size=(areas, questionnaire_items))
    latent = area_shift[area] + loading * trait[:, None] + rng.normal(0, 1, size=(n, questionnaire_items))
    questionnaire = _ordinal(latent, categories)

    battery_shift = rng.normal(battery_level, area_sd, size=(areas, len(BATTERY)))
    battery_latent = battery_shift[area] + loading * trait[:, None] + rng.normal(size=(n, len(BATTERY)))
    battery = _ordinal(battery_latent + effect * treated[:, None], 5)

    for who in fabricators:
        own = np.flatnonzero(interviewer == who)
        invented = own[rng.random(len(own)) < fabricated_share]
        treated[invented] = rng.random(len(invented)) >= fabricated_control
        battery[invented] = np.where(treated[invented, None], [4, 1, 0, 4], [4, 0, 0, 4])

    for who in copiers:
        own = np.flatnonzero(interviewer == who)
        for position in range(1, len(own)):
            if rng.random() < copied_share:
                source = own[rng.integers(0, position)]
                row = questionnaire[source].copy()
                change = rng.random(questionnaire_items) < 0.05
                row[change] = rng.integers(0, categories, size=change.sum())
                questionnaire[own[position]] = row

    frame = pd.DataFrame(questionnaire, columns=[f"q{i:02d}" for i in range(questionnaire_items)])
    frame.insert(0, "interviewer", interviewer)
    frame.insert(1, "area", area)
    frame.insert(2, "treated", treated.astype(int))
    frame[BATTERY] = battery
    frame["support"] = battery.mean(axis=1) / 4
    return frame


def questionnaire_items(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("q")]
