"""Checks on when interviews happened: overlapping clocks, crowded days and short interviews."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from interviewer_checks._util import bonferroni


def _minutes(start: pd.Series, end: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(start):
        return (end - start) / pd.Timedelta(minutes=1)
    return end.astype(np.float64) - start.astype(np.float64)


def _day_columns(df: pd.DataFrame, start: Hashable, date: Hashable | Sequence[Hashable] | None) -> pd.DataFrame:
    if date is None:
        if pd.api.types.is_datetime64_any_dtype(df[start]):
            return pd.DataFrame({"_day": df[start].dt.normalize()}, index=df.index)
        return pd.DataFrame({"_day": np.zeros(len(df))}, index=df.index)
    columns = [date] if isinstance(date, str) or not isinstance(date, Sequence) else list(date)
    return df[columns]


def clock_flags(
    df: pd.DataFrame,
    interviewer: Hashable,
    start: Hashable,
    *,
    end: Hashable | None = None,
    duration: Hashable | None = None,
    date: Hashable | Sequence[Hashable] | None = None,
    min_minutes: float | None = None,
    max_minutes: float | None = None,
) -> pd.DataFrame:
    """Flag each interview whose recorded times cannot all be true, or that was implausibly short.

    Mechanism: interviews are grouped by interviewer and working day and sorted by start time. An interview
    ``overlaps`` when it starts before the latest end time among that interviewer's earlier interviews that day. One
    person cannot hold two interviews at once, so an overlap means either that the times were not written down as the
    interviews happened or that more than one person worked under the interviewer's id. An interview is ``short`` when
    it lasted less than ``min_minutes``.

    Times may be datetimes or plain numbers in minutes (e.g. minutes since midnight). The end of an interview comes
    from ``end`` or, failing that, from ``start + duration`` (in minutes). Without either, durations are unknown and
    only the day counts are meaningful. An interview is ``timed`` when its duration is positive, finite and at most
    ``max_minutes``; untimed interviews (no times, an end before the start, an implausibly long interview that is
    likely a keying error) take no part in the overlap comparison.

    Args:
        df: One row per interview.
        interviewer: Interviewer column.
        start: Start time column (datetime or minutes).
        end: End time column, of the same kind as ``start``.
        duration: Duration column in minutes, used when ``end`` is not given.
        date: Column, or columns, that identify one working day (e.g. ``["village", "day"]``). Defaults to the
            calendar date of a datetime ``start``; with numeric times and no ``date``, each interviewer's interviews
            are treated as one day.
        min_minutes: Interviews shorter than this are ``short``. ``None`` turns the check off.
        max_minutes: Longer interviews count as untimed. ``None`` keeps them.

    Returns:
        A frame with the index of ``df`` and columns ``minutes``, ``timed``, ``follows`` (a timed interview with an
        earlier timed interview by the same interviewer that day, so one that could overlap), ``overlap``, ``short``
        and ``flagged`` (overlap or short).
    """
    if end is not None:
        minutes = _minutes(df[start], df[end])
    elif duration is not None:
        minutes = df[duration].astype(np.float64)
    else:
        minutes = pd.Series(np.nan, index=df.index)
    minutes = minutes.astype(np.float64)
    timed = np.isfinite(minutes) & (minutes > 0) & df[start].notna()
    if max_minutes is not None:
        timed &= minutes <= max_minutes

    days = _day_columns(df, start, date)
    keys = pd.concat([df[[interviewer]], days], axis=1)
    keys.columns = [f"_key{i}" for i in range(keys.shape[1])]
    key_columns = list(keys.columns)
    frame = keys.assign(_start=df[start], _minutes=minutes)[timed & keys.notna().all(axis=1)]
    rows = np.flatnonzero((timed & keys.notna().all(axis=1)).to_numpy())
    frame.index = rows
    if pd.api.types.is_datetime64_any_dtype(df[start]):
        frame["_end"] = frame["_start"] + pd.to_timedelta(frame["_minutes"], unit="min")
    else:
        frame["_end"] = frame["_start"].astype(np.float64) + frame["_minutes"]
    frame = frame.sort_values([*key_columns, "_start"], kind="mergesort")
    previous_end = frame.groupby(key_columns, sort=False)["_end"].transform(lambda s: s.cummax().shift())

    result = pd.DataFrame(index=df.index)
    result["minutes"] = minutes
    result["timed"] = timed.to_numpy()
    follows = np.zeros(len(df), dtype=bool)
    overlap = np.zeros(len(df), dtype=bool)
    follows[frame.index] = previous_end.notna().to_numpy()
    overlap[frame.index] = (frame["_start"] < previous_end).to_numpy()
    result["follows"] = follows
    result["overlap"] = overlap
    result["short"] = timed & (minutes < min_minutes) if min_minutes is not None else False
    result["flagged"] = result["overlap"] | result["short"]
    return result


def interview_clock(
    df: pd.DataFrame,
    interviewer: Hashable,
    start: Hashable,
    *,
    end: Hashable | None = None,
    duration: Hashable | None = None,
    date: Hashable | Sequence[Hashable] | None = None,
    min_minutes: float | None = None,
    max_minutes: float | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per interviewer: how often their recorded interview times overlap, how crowded their days are, how short.

    Mechanism: :func:`clock_flags` marks each interview that starts before the same interviewer's previous
    interview that day has ended. ``overlap_share`` is the share of an interviewer's ``follows`` interviews (those
    with an earlier interview that day, the only ones that can overlap) that do. Honest paper schedules produce a
    few overlaps through rounding and slips of the pen, so the flag is relative: an exact one-sided binomial test of
    the interviewer's overlap count against the survey-wide overlap rate, with a Bonferroni correction. The null is
    that every interviewer makes clock errors at the same rate. It says nothing when the whole survey's clocks are
    wrong, and a flagged interviewer's times are impossible, which is not the same as invented answers.

    ``max_per_day`` is the most interviews the interviewer logged on one working day, ``median_minutes`` the median
    duration of timed interviews, and ``short_share`` the share of timed interviews below ``min_minutes``. These are
    descriptive: what counts as too many or too short depends on the questionnaire.

    Args:
        df: One row per interview.
        interviewer: Interviewer column.
        start: Start time column (datetime or minutes).
        end: End time column.
        duration: Duration column in minutes, used when ``end`` is not given.
        date: Column, or columns, that identify one working day; see :func:`clock_flags`.
        min_minutes: Floor below which an interview is ``short``.
        max_minutes: Longer interviews count as untimed.
        alpha: Family-wise error rate for ``flagged``.

    Returns:
        One row per interviewer with ``interviews``, ``timed``, ``follows``, ``overlaps``, ``overlap_share``,
        ``max_per_day``, ``median_minutes``, ``short_share``, ``p_value``, ``p_adjusted`` and ``flagged``. ``attrs``
        holds the survey-wide ``overlap_share`` (overlaps over follows), ``timed_share`` and ``short_share``. The
        interview-level flags come from :func:`clock_flags`.
    """
    flags = clock_flags(
        df,
        interviewer,
        start,
        end=end,
        duration=duration,
        date=date,
        min_minutes=min_minutes,
        max_minutes=max_minutes,
    )
    days = _day_columns(df, start, date)
    dated = days.notna().all(axis=1)
    has_days = date is not None or pd.api.types.is_datetime64_any_dtype(df[start])
    per_day = pd.concat([df[[interviewer]], days], axis=1)[dated].value_counts()
    max_per_day = per_day.groupby(level=0).max() if has_days else pd.Series(dtype=np.float64)

    frame = flags.assign(**{"_who": df[interviewer].to_numpy()})
    timed_minutes = frame["minutes"].where(frame["timed"])
    frame["_timed_minutes"] = timed_minutes
    frame["_short_timed"] = frame["short"].where(frame["timed"]).astype(np.float64)
    table = frame.groupby("_who").agg(
        interviews=("timed", "size"),
        timed=("timed", "sum"),
        follows=("follows", "sum"),
        overlaps=("overlap", "sum"),
        median_minutes=("_timed_minutes", "median"),
        short_share=("_short_timed", "mean"),
    )
    table.index.name = interviewer
    table = table.reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        table["overlap_share"] = table["overlaps"] / table["follows"].where(table["follows"] > 0)
    table["max_per_day"] = table[interviewer].map(max_per_day).astype(np.float64)
    if min_minutes is None:
        table["short_share"] = np.nan

    follows, overlaps = int(table["follows"].sum()), int(table["overlaps"].sum())
    rate = overlaps / follows if follows else np.nan
    p_values = np.array(
        [
            stats.binomtest(int(k), int(n), rate, alternative="greater").pvalue if n > 0 and 0 < rate < 1 else np.nan
            for k, n in zip(table["overlaps"], table["follows"])
        ]
    )
    table["p_value"] = p_values
    table["p_adjusted"] = bonferroni(p_values)
    table["flagged"] = table["p_adjusted"] < alpha
    table = table[
        [interviewer, "interviews", "timed", "follows", "overlaps", "overlap_share", "max_per_day"]
        + ["median_minutes", "short_share", "p_value", "p_adjusted", "flagged"]
    ]
    timed = flags["timed"]
    table.attrs.update(
        overlap_share=float(rate),
        timed_share=float(timed.mean()),
        short_share=float(flags.loc[timed, "short"].mean()) if min_minutes is not None and timed.any() else np.nan,
    )
    return table
