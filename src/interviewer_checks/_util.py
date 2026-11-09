from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def labels(df: pd.DataFrame, column: Hashable | None) -> tuple[np.ndarray, np.ndarray]:
    """Integer codes for a grouping column and the values they stand for.

    A missing column (``None``) puts every row in one group, so checks that condition on area still run when the
    data have no areas.
    """
    if column is None:
        return np.zeros(len(df), dtype=np.int64), np.array([None], dtype=object)
    if df[column].isna().any():
        raise ValueError(f"Column {column!r} has missing values; drop or fill them first")
    codes, uniques = pd.factorize(df[column], sort=True)
    return codes.astype(np.int64), np.asarray(uniques, dtype=object)


def encode_items(df: pd.DataFrame, items: Sequence[Hashable]) -> np.ndarray:
    """Answers as integer codes per item, with -1 for a missing answer."""
    if len(items) == 0:
        raise ValueError("Pass at least one item")
    missing = [item for item in items if item not in df.columns]
    if missing:
        raise KeyError(f"Items not in the data: {missing}")
    return np.column_stack([pd.factorize(df[item], sort=True)[0] for item in items]).astype(np.int64)


def one_hot(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One column per (item, category), plus the answered-item indicator and each column's item.

    Two interviews agree on an item exactly when they share a 1 in one of its columns, so the agreement counts for
    every pair are one matrix product. float32 keeps the products small; the counts are integers well below 2**24,
    so they are exact.
    """
    n_rows, n_items = codes.shape
    sizes = codes.max(axis=0) + 1
    sizes = np.maximum(sizes, 1)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    indicators = np.zeros((n_rows, int(sizes.sum())), dtype=np.float32)
    rows, cols = np.nonzero(codes >= 0)
    indicators[rows, offsets[cols] + codes[rows, cols]] = 1.0
    answered = (codes >= 0).astype(np.float32)
    item_of_column = np.repeat(np.arange(n_items), sizes)
    return indicators, answered, item_of_column


def poisson_binomial_sf(probabilities: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """P(X >= observed) for each row, where X is a sum of independent Bernoulli draws with that row's probabilities.

    Exact, by building the distribution of X one draw at a time. Rows are evaluated together, so the loop runs over
    columns only.
    """
    probabilities = np.atleast_2d(np.asarray(probabilities, dtype=np.float64))
    observed = np.atleast_1d(np.asarray(observed, dtype=np.int64))
    n_rows, n_draws = probabilities.shape
    dist = np.zeros((n_rows, n_draws + 1))
    dist[:, 0] = 1.0
    for column in range(n_draws):
        q = probabilities[:, column : column + 1]
        shifted = dist[:, :-1] * q
        dist *= 1.0 - q
        dist[:, 1:] += shifted
    tail = np.cumsum(dist[:, ::-1], axis=1)[:, ::-1]
    tail = np.concatenate([tail, np.zeros((n_rows, 1))], axis=1)
    index = np.clip(observed, 0, n_draws + 1)
    return np.minimum(tail[np.arange(n_rows), index], 1.0)


def hypergeometric_sum_sf(cells: Sequence[tuple[int, int, int]], observed: int) -> float:
    """P(X >= observed) for X a sum of independent hypergeometric counts, one per (population, successes, draws).

    This is the exact null for "how many of an interviewer's interviews have a property" when interviewer labels are
    shuffled within each area: in an area with N interviews, K of them with the property, and n by this interviewer,
    the interviewer's count is hypergeometric, and areas are shuffled independently.
    """
    dist = np.ones(1)
    for population, successes, draws in cells:
        support = np.arange(0, draws + 1)
        dist = np.convolve(dist, stats.hypergeom.pmf(support, population, successes, draws))
    return float(min(dist[max(observed, 0) :].sum(), 1.0)) if observed < len(dist) else 0.0


def bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Multiply by the number of finite p-values and cap at one; missing p-values stay missing."""
    p_values = np.asarray(p_values, dtype=np.float64)
    tested = np.isfinite(p_values).sum()
    return np.minimum(p_values * max(tested, 1), 1.0)


def permutation_p(null: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Share of null draws at least as large as the observed value, counting the observed value as one draw."""
    exceed = (null >= observed[np.newaxis, :] - 1e-12).sum(axis=0)
    p_values = (1.0 + exceed) / (1.0 + null.shape[0])
    return np.where(np.isfinite(observed), p_values, np.nan)


def shuffle_within(cells: np.ndarray, values: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    """``draws`` copies of ``values``, each shuffled independently within every cell."""
    out = np.empty((draws, len(values)), dtype=values.dtype)
    for cell in np.unique(cells):
        members = np.flatnonzero(cells == cell)
        out[:, members] = rng.permuted(np.broadcast_to(values[members], (draws, len(members))), axis=1)
    return out


def straightlined(codes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interviews with at least two answered items that all carry the same value."""
    answered = codes >= 0
    low = np.where(answered, values, np.inf).min(axis=1)
    high = np.where(answered, values, -np.inf).max(axis=1)
    return (answered.sum(axis=1) >= 2) & (low == high)


def numeric_values(df: pd.DataFrame, items: Sequence[Hashable]) -> np.ndarray:
    """Answers as floats when every item is numeric, otherwise their integer codes.

    Straight-lining compares values across items, which only means "same answer" when the items share a scale; for
    non-numeric items the same category label is the closest analog.
    """
    frame = df[list(items)]
    if all(pd.api.types.is_numeric_dtype(frame[item]) for item in items):
        return frame.to_numpy(dtype=np.float64)
    stacked = pd.Series(frame.to_numpy().ravel())
    codes = pd.factorize(stacked)[0].astype(np.float64)
    codes[codes < 0] = np.nan
    return codes.reshape(frame.shape)


def answer_strings(df: pd.DataFrame, items: Sequence[Hashable]) -> np.ndarray:
    """Each interview's answers joined into one string, e.g. ``4-0-0-4``; a missing answer prints as ``.``."""

    def show(value: object) -> str:
        if pd.isna(value):
            return "."
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    frame = df[list(items)]
    return np.array(["-".join(show(value) for value in row) for row in frame.itertuples(index=False)], dtype=object)
