"""Run every check the data allow and collect one row per interviewer."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from interviewer_checks.clock import interview_clock
from interviewer_checks.experiments import assignment_balance, cluster_influence, effect_heterogeneity
from interviewer_checks.responses import interviewer_similarity, pattern_repetition, percent_match, straightlining


@dataclass(frozen=True)
class Screen:
    """Result of :func:`screen`.

    Attributes:
        table: One row per interviewer, most-flagged first.
        summary: A short plain-text account of what each check found.
        results: The full output of each check that ran, keyed by function name.
    """

    table: pd.DataFrame
    summary: str
    results: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.summary


def _names(table: pd.DataFrame, key: Hashable, mask: pd.Series) -> str:
    flagged = table.loc[mask, key].tolist()
    return ", ".join(str(value) for value in flagged) if flagged else "none"


def screen(
    df: pd.DataFrame,
    interviewer: Hashable,
    *,
    items: Sequence[Hashable] | None = None,
    pattern_items: Sequence[Hashable] | None = None,
    area: Hashable | None = None,
    treatment: Hashable | None = None,
    outcome: Hashable | None = None,
    start: Hashable | None = None,
    end: Hashable | None = None,
    duration: Hashable | None = None,
    date: Hashable | Sequence[Hashable] | None = None,
    min_minutes: float | None = None,
    max_minutes: float | None = None,
    n_permutations: int = 9999,
    min_respondents: int = 20,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> Screen:
    """Run the applicable checks on one-row-per-interview data and tabulate them by interviewer.

    Which checks run depends on what is passed:

    - ``treatment`` and ``outcome``: :func:`effect_heterogeneity` (interviewer-specific effects),
      :func:`cluster_influence` on the difference in means, and :func:`assignment_balance`. For an outcome made of
      several items, pass their mean: with every respondent answering every item, the difference in respondent means
      equals the long-format difference and its row-level standard error equals the respondent-clustered one.
    - ``treatment`` alone: :func:`assignment_balance`.
    - ``items``: :func:`interviewer_similarity` (flags interviewers whose interviews have close twins among their own
      work) and :func:`percent_match`, whose per-interviewer counts (``near_duplicates``, ``exact_duplicates``) are
      reported but do not flag, because its chance benchmark assumes independent items.
    - ``pattern_items`` (defaults to ``items``): :func:`pattern_repetition` (stratified by ``treatment`` when given)
      and :func:`straightlining`. Use a short battery here (the outcome items of an experiment); full strings over a
      long questionnaire almost never repeat.
    - ``start`` (with ``end`` or ``duration``, and ``date``): :func:`interview_clock`, which flags interviewers whose
      interviews overlap their own previous interview more often than the survey-wide rate, and reports crowded days
      and short interviews.

    The checks are deliberately generic. None looks for a particular string or pattern; together they ask whether an
    estimate depends on a few interviewers, whether interviewers differ in their treatment effects more than
    sampling noise allows, whether randomization was followed, and whether answers repeat within interviewers more
    than local opinion explains.

    Args:
        df: One row per interview.
        interviewer: Interviewer column.
        items: Questionnaire columns for the near-duplicate checks.
        pattern_items: Columns for answer strings and straight-lining; defaults to ``items``.
        area: Area column (sampling point, district). The null distributions are built within it.
        treatment: 0/1 treatment column of a survey experiment.
        outcome: Numeric outcome column of the experiment.
        start: Interview start time (datetime or minutes).
        end: Interview end time.
        duration: Interview length in minutes, used when ``end`` is not given.
        date: Column(s) identifying a working day; see :func:`clock_flags`.
        min_minutes: Floor below which an interview counts as short.
        max_minutes: Longer interviews count as untimed.
        n_permutations: Shuffles and redraws for the permutation checks.
        min_respondents: Smallest interviewer that enters the heterogeneity test.
        alpha: Family-wise error rate behind every flag.
        seed: Seed for the permutation checks.

    Returns:
        A :class:`Screen` with the per-interviewer ``table`` (one ``flag_*`` column per check, ``flags`` naming
        them and ``n_flags`` counting them), a text ``summary`` and the full ``results``.
    """
    pattern_items = pattern_items if pattern_items is not None else items
    key = interviewer
    table = df.groupby(key).size().rename("interviews").reset_index()
    results: dict[str, pd.DataFrame] = {}
    lines = [f"{len(table)} interviewers, {len(df):,} interviews."]

    if area is not None:
        per_area = df.groupby(area)[key].nunique()
        single = np.mean(per_area == 1)
        line = f"Areas ({area}): {len(per_area)}, {single:.0%} worked by a single interviewer."
        if single > 0:
            line += " There interviewer and area cannot be told apart, and the within-area checks skip them."
        if single > 0.5:
            line += " Most areas are confounded with interviewers; consider a coarser area."
        lines.append(line)

    def merge(result: pd.DataFrame, columns: dict[str, str]) -> None:
        nonlocal table
        table = table.merge(result[[key, *columns]].rename(columns=columns), on=key, how="left")

    if treatment is not None and outcome is not None:
        effects = effect_heterogeneity(df, outcome, treatment, key, min_respondents=min_respondents, alpha=alpha)
        results["effect_heterogeneity"] = effects
        merge(
            effects,
            {"effect": "effect", "z": "effect_z", "flagged": "flag_effect", "zero_variance": "flag_zero_variance"},
        )
        attrs = effects.attrs
        lines.append(
            f"Pooled effect {attrs['effect']:.3f} (SE {attrs['se']:.3f}; {attrs['se_cluster']:.3f} clustered by "
            f"interviewer). Heterogeneity across interviewers: chi2 = {attrs['chi2']:.1f} on {attrs['df']} df, "
            f"p = {attrs['p_value']:.3g}. Interviewer effects off the pooled one: "
            f"{_names(effects, key, effects['flagged'])}. No variation in the outcome: "
            f"{', '.join(map(str, attrs['zero_variance'])) or 'none'}."
        )

        def difference(frame: pd.DataFrame) -> float:
            treated = frame[treatment].astype(bool)
            return float(frame.loc[treated, outcome].mean() - frame.loc[~treated, outcome].mean())

        influence = cluster_influence(df, difference, key)
        results["cluster_influence"] = influence
        merge(influence, {"shift": "ate_shift", "rank": "influence_rank"})
        top = influence.head(3)
        lines.append(
            "Leaving one interviewer out moves the effect most for "
            + ", ".join(
                f"{row[key]} ({row['shift']:+.4f}, {row['shift_in_se']:+.1f} jackknife SE)" for _, row in top.iterrows()
            )
            + "."
        )

    if treatment is not None:
        balance = assignment_balance(df, treatment, key, alpha=alpha)
        results["assignment_balance"] = balance
        merge(balance, {"share_treated": "share_treated", "flagged": "flag_balance"})
        lines.append(
            f"Assignment: {balance.attrs['share_treated']:.1%} treated overall; interviewer x treatment chi2 = "
            f"{balance.attrs['chi2']:.1f} on {balance.attrs['df']} df, p = {balance.attrs['p_value']:.3g}. "
            f"Lopsided interviewers: {_names(balance, key, balance['flagged'])}."
        )

    if pattern_items is not None:
        repetition = pattern_repetition(
            df, pattern_items, key, area, strata=treatment, n_simulations=n_permutations, alpha=alpha, seed=seed
        )
        results["pattern_repetition"] = repetition
        merge(
            repetition,
            {"repeated_share": "repeated_share", "expected_share": "repeated_expected", "flagged": "flag_repetition"},
        )
        lines.append(
            f"Repeated answer strings beyond what the area produces: {_names(repetition, key, repetition['flagged'])}."
        )
        flat = straightlining(df, pattern_items, key, area, alpha=alpha)
        results["straightlining"] = flat
        merge(flat, {"share": "straightline_share", "flagged": "flag_straightlining"})
        lines.append(
            f"Straight-lining ({flat.attrs['share']:.1%} of interviews overall) above the area rate: "
            f"{_names(flat, key, flat['flagged'])}."
        )

    if items is not None:
        matches = percent_match(df, items, area=area, interviewer=key, alpha=alpha)
        results["percent_match"] = matches
        per = matches.assign(**{key: df[key].to_numpy()}).groupby(key)
        counts = per.agg(
            near_duplicates=("flagged", "sum"), exact_duplicates=("exact_duplicates", lambda s: int((s > 0).sum()))
        )
        counts = counts.reset_index()
        table = table.merge(counts, on=key, how="left")
        lines.append(
            f"Near duplicates over {matches.attrs['items']} items: {matches.attrs['share_at_85']:.1%} of interviews "
            f"have a best match >= 0.85 and {matches.attrs['exact_duplicate_pairs']} pairs match exactly; "
            f"{matches.attrs['flagged']} interviews match their best partner beyond what independent items would give "
            "(a screen, not a flag: real items correlate)."
        )
        similarity = interviewer_similarity(df, items, key, area, n_permutations=n_permutations, alpha=alpha, seed=seed)
        results["interviewer_similarity"] = similarity
        merge(similarity, {"excess": "similarity_excess", "flagged": "flag_similarity"})
        attrs = similarity.attrs
        lines.append(
            f"Closest same-interviewer twin vs. shuffled labels within area: {attrs['excess']:+.3f} agreement "
            f"(permutation p = {attrs['p_value']:.3g}); interviewers whose interviews resemble each other beyond "
            f"chance: {_names(similarity, key, similarity['flagged'])}."
        )

    if start is not None:
        times = interview_clock(
            df,
            key,
            start,
            end=end,
            duration=duration,
            date=date,
            min_minutes=min_minutes,
            max_minutes=max_minutes,
            alpha=alpha,
        )
        results["interview_clock"] = times
        merge(
            times,
            {
                "overlap_share": "overlap_share",
                "max_per_day": "max_per_day",
                "short_share": "short_share",
                "flagged": "flag_clock",
            },
        )
        line = (
            f"Clock: {times.attrs['overlap_share']:.1%} of interviews that follow another the same day start before it "
            f"ended; more often than that rate: {_names(times, key, times['flagged'])}."
        )
        if min_minutes is not None:
            line += f" Shorter than {min_minutes:g} minutes: {times.attrs['short_share']:.1%} of timed interviews."
        lines.append(line)

    flag_columns = [column for column in table.columns if column.startswith("flag_")]
    for column in flag_columns:
        table[column] = table[column].eq(True)
    table["flags"] = (
        table[flag_columns].apply(
            lambda row: ", ".join(column.removeprefix("flag_") for column in flag_columns if row[column]), axis=1
        )
        if flag_columns
        else ""
    )
    table["n_flags"] = table[flag_columns].sum(axis=1) if flag_columns else 0
    table = table.sort_values(["n_flags", key], ascending=[False, True], ignore_index=True)
    several = table.loc[table["n_flags"] >= 2, key].tolist()
    lines.append(f"Flagged by two or more checks: {', '.join(map(str, several)) or 'none'}.")
    return Screen(table=table, summary="\n".join(lines), results=results)
