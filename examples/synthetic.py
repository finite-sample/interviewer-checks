"""Worked example from the README: a simulated survey experiment with one interviewer inventing answers."""

import numpy as np
import pandas as pd

from interviewer_checks import screen

rng = np.random.default_rng(1)
areas, per_area, per_interviewer = 20, 3, 120
interviewer = np.repeat(np.arange(areas * per_area), per_interviewer)
area = interviewer // per_area
treated = rng.integers(0, 2, interviewer.size)

# Four 0-4 items driven by local opinion, a respondent trait, and a small negative treatment effect.
latent = rng.normal(1.5, 0.5, (areas, 4))[area] + 0.6 * rng.normal(size=(interviewer.size, 1))
latent += rng.normal(size=(interviewer.size, 4)) - 0.15 * treated[:, None]
answers = np.searchsorted([-1.5, -0.5, 0.5, 1.5], latent).astype(float)

# Interviewer 12 invents 60% of interviews: mostly "control", answered 4-0-0-4, and 4-1-0-4 when "treated".
invented = (interviewer == 12) & (rng.random(interviewer.size) < 0.6)
treated[invented] = rng.random(invented.sum()) < 0.15
answers[invented] = np.where(treated[invented, None] == 1, [4, 1, 0, 4], [4, 0, 0, 4])

df = pd.DataFrame(answers, columns=["p1", "p2", "p3", "p4"])
df["interviewer"], df["area"], df["treated"] = interviewer, area, treated
df["support"] = df[["p1", "p2", "p3", "p4"]].mean(axis=1) / 4

result = screen(
    df,
    "interviewer",
    pattern_items=["p1", "p2", "p3", "p4"],
    area="area",
    treatment="treated",
    outcome="support",
)
print(result.summary)
print(result.table.head(4)[["interviewer", "effect_z", "share_treated", "repeated_share", "flags"]])
