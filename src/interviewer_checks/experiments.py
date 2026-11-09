"""Checks on an estimate and on a randomized treatment, cluster by cluster."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from interviewer_checks._util import bonferroni, labels, permutation_p, shuffle_within


def cluster_influence(
    df: pd.DataFrame, estimate_fn: Callable[[pd.DataFrame], float], cluster: Hashable
) -> pd.DataFrame:
    """Re-estimate with each cluster left out, and report how far the estimate moves.

    Mechanism: ``estimate_fn`` is applied to the full data and then to the data without each cluster in turn
    (a delete-one-cluster jackknife). ``shift`` is the leave-out estimate minus the full estimate, so a cluster whose
    removal lowers the estimate has a negative shift. An estimate that rests on a few clusters shows up as a few
    shifts far larger than the rest. ``shift_in_se`` divides each shift by the jackknife standard error of the
    estimate, ``sqrt((G - 1) / G * sum((leave_out - mean(leave_out))**2))`` over the G clusters, which puts the
    shifts on the scale of the estimate's own sampling noise. Nothing here is a test: a cluster can move the estimate
    because it is large or because its data are wrong, and the table only says which clusters to look at.

    Args:
        df: The data, at whatever level ``estimate_fn`` expects.
        estimate_fn: Maps a data frame to a single number, e.g. a difference in means.
        cluster: Column naming the cluster (interviewer, area) to leave out.

    Returns:
        One row per cluster, sorted by the size of the shift, with ``rows``, ``estimate_without``, ``shift``,
        ``shift_in_se`` and ``rank``. ``attrs`` holds the full-data ``estimate`` and the ``jackknife_se``.
    """
    codes, values = labels(df, cluster)
    full = float(estimate_fn(df))
    leave_out = np.array([float(estimate_fn(df.loc[codes != code])) for code in range(len(values))])
    groups = len(values)
    jackknife_se = float(np.sqrt((groups - 1) / groups * np.sum((leave_out - leave_out.mean()) ** 2)))
    shift = leave_out - full
    table = pd.DataFrame(
        {
            cluster: values,
            "rows": np.bincount(codes, minlength=groups),
            "estimate_without": leave_out,
            "shift": shift,
            "shift_in_se": shift / jackknife_se if jackknife_se > 0 else np.nan,
        }
    )
    table = table.sort_values("shift", key=np.abs, ascending=False, ignore_index=True)
    table["rank"] = np.arange(1, groups + 1)
    table.attrs.update(estimate=full, jackknife_se=jackknife_se)
    return table


def _difference_in_means(y: np.ndarray, treated: np.ndarray, units: np.ndarray) -> tuple[float, float]:
    """Difference in means and its standard error clustered on ``units``.

    The difference in means is a sum of per-row scores (each row's deviation from its arm mean, divided by its arm's
    size, with a minus sign in control). Summing the scores within a unit and squaring gives the cluster-robust
    (CR0) variance, which lets the several answers one respondent gives move together.
    """
    n_treated, n_control = treated.sum(), (~treated).sum()
    if n_treated == 0 or n_control == 0:
        return np.nan, np.nan
    mean_treated, mean_control = y[treated].mean(), y[~treated].mean()
    residual = y - np.where(treated, mean_treated, mean_control)
    score = residual * np.where(treated, 1.0 / n_treated, -1.0 / n_control)
    by_unit = np.bincount(pd.factorize(units)[0], weights=score)
    return float(mean_treated - mean_control), float(np.sqrt(np.sum(by_unit**2)))


def _binary(series: pd.Series, name: Hashable) -> np.ndarray:
    observed = set(pd.unique(series.dropna()))
    if series.isna().any() or not observed <= {0, 1, True, False}:
        raise ValueError(f"Column {name!r} must be 0/1 or boolean with no missing values")
    return series.to_numpy().astype(bool)


# Below this many units in some tested cluster's smaller arm, "auto" judges Q by permutation. Q sums squared
# z-scores whose standard errors come from the data, so each term has a heavier tail than a squared normal and its
# mean is inflated by roughly 1/n; the excess adds up over clusters. In simulation (normal outcomes, homogeneous
# effect, 3,000-4,000 replications) the chi-square reference rejects at 5% in 15% of runs with 12 clusters of 10
# units per arm, 35% with 60 such clusters, and 80% with 60 clusters of 5; at 50 per arm it is still 7-9% for 60-120
# clusters, and at 100 per arm 6-8% for 12-300 clusters. A threshold of 10 would leave the chi-square path badly
# anti-conservative, so the switch sits where the inflation becomes small. The permutation costs well under a second
# on surveys of a few thousand respondents.
CHI2_MIN_ARM = 100


def _cluster_effects(
    sums: np.ndarray, counts: np.ndarray, starts: np.ndarray, treated: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cluster difference in means and its unit-clustered SE, plus the pooled difference, for many labelings.

    ``sums`` and ``counts`` are each unit's outcome total and row count, with units sorted by cluster and cluster
    ``g`` starting at ``starts[g]``; ``treated`` is (draws x units). Because a unit sits wholly in one arm, its summed
    score is ``(sum - count * arm_mean) / arm_rows``, which is what :func:`_difference_in_means` squares and adds.
    """
    rows = np.add.reduceat(counts, starts)
    total = np.add.reduceat(sums, starts)
    rows_treated = np.add.reduceat(treated * counts, starts, axis=1)
    sum_treated = np.add.reduceat(treated * sums, starts, axis=1)
    rows_control = rows - rows_treated
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_treated = sum_treated / rows_treated
        mean_control = (total - sum_treated) / rows_control
        cluster_of_unit = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, len(sums))))
        arm_mean = np.where(treated, mean_treated[:, cluster_of_unit], mean_control[:, cluster_of_unit])
        arm_rows = np.where(treated, rows_treated[:, cluster_of_unit], rows_control[:, cluster_of_unit])
        score = (sums - counts * arm_mean) / arm_rows
        se = np.sqrt(np.add.reduceat(score**2, starts, axis=1))
        pooled = sum_treated.sum(axis=1) / rows_treated.sum(axis=1) - (total.sum() - sum_treated.sum(axis=1)) / (
            rows.sum() - rows_treated.sum(axis=1)
        )
    return mean_treated - mean_control, se, pooled


def _q_statistic(effects: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Cochran's Q along the last axis, with its inverse-variance weighted centre; a zero SE makes Q infinite."""
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = 1.0 / se**2
        centre = np.sum(weights * effects, axis=-1) / np.sum(weights, axis=-1)
        q = np.sum(weights * (effects - centre[..., np.newaxis]) ** 2, axis=-1)
    return np.where(np.all(se > 0, axis=-1), q, np.inf), centre


def _permutation_null(
    y: np.ndarray,
    treated: np.ndarray,
    units: np.ndarray,
    codes: np.ndarray,
    centre: float,
    tested: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Null draws of Q and of every tested cluster's |z|, shuffling unit-level labels within each cluster."""
    frame = pd.DataFrame({"cluster": codes, "unit": pd.factorize(units)[0], "treated": treated, "y": y})
    frame["y"] -= centre * frame["treated"]
    per_unit = frame.groupby(["cluster", "unit"], sort=True).agg(
        sums=("y", "sum"), counts=("y", "size"), treated=("treated", "first"), arms=("treated", "nunique")
    )
    mixed = int((per_unit["arms"] > 1).sum())
    if mixed:
        raise ValueError(
            f"Treatment varies within {mixed} respondent(s); the permutation test shuffles whole respondents, so "
            "drop them or use method='chi2'"
        )
    cluster_of_unit = per_unit.index.get_level_values("cluster").to_numpy()
    starts = np.flatnonzero(np.r_[True, cluster_of_unit[1:] != cluster_of_unit[:-1]])
    sums, counts = per_unit["sums"].to_numpy(), per_unit["counts"].to_numpy(dtype=np.float64)
    labels_now = per_unit["treated"].to_numpy()
    null_q = np.empty(n_permutations)
    null_z = np.empty((n_permutations, int(tested.sum())))
    batch = max(1, 4_000_000 // len(sums))
    for start in range(0, n_permutations, batch):
        draws = min(batch, n_permutations - start)
        shuffled = shuffle_within(cluster_of_unit, labels_now, draws, rng)
        effects, se, pooled = _cluster_effects(sums, counts, starts, shuffled)
        effects, se = effects[:, tested], se[:, tested]
        null_q[start : start + draws] = _q_statistic(effects, se)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.abs(effects - pooled[:, np.newaxis]) / se
        null_z[start : start + draws] = np.where(se > 0, z, np.inf)
    return null_q, null_z


def effect_heterogeneity(
    df: pd.DataFrame,
    outcome: Hashable,
    treatment: Hashable,
    cluster: Hashable,
    *,
    respondent: Hashable | None = None,
    min_respondents: int = 20,
    alpha: float = 0.05,
    method: Literal["auto", "chi2", "permutation"] = "auto",
    n_permutations: int = 9999,
    seed: int | None = None,
) -> pd.DataFrame:
    """Each cluster's own treatment effect, compared with the pooled effect.

    Mechanism: within each cluster (usually an interviewer), the effect is the difference between the mean outcome
    of treated and control rows. Its standard error is clustered on ``respondent`` when the data are long (several
    outcome rows per respondent), otherwise each row is its own unit. ``z`` is ``(effect - pooled) / se``: how many of
    its own standard errors a cluster's effect sits from the effect in the whole sample.

    The heterogeneity statistic is Cochran's Q, the standard one from meta-analysis: weight each tested cluster's
    effect by its inverse variance, take the weighted mean (``centre``), and sum the weighted squared deviations from
    it. (Summing the squared ``z`` against the pooled difference in means gives a slightly larger number, because the
    weighted mean is the centre that minimizes the sum.) Q is judged in one of two ways:

    - ``"chi2"``: against a chi-square with one fewer degree of freedom than tested clusters, and each ``z`` against
      a normal with a Bonferroni correction. Both treat the standard errors as known, which is close to true only
      when every arm of every tested cluster has many units; with few, Q and the largest ``|z|`` are too big too often.
    - ``"permutation"``: subtract ``centre`` times the treatment indicator from the outcome, so that every cluster
      has an effect near zero (the null of a common effect, with the common effect estimated), then shuffle the
      treatment labels of units (respondents, or rows) within each cluster, which keeps each cluster's number of
      treated and control units. Recompute Q and every tested cluster's ``|z|`` on each of ``n_permutations``
      shuffles. The p-value of Q is ``(1 + #{null Q >= observed}) / (1 + n_permutations)``. A cluster's ``p_value`` is
      the same share for its own ``|z|``, and ``p_adjusted`` compares its ``|z|`` with the largest ``|z|`` across
      tested clusters in each shuffle (Westfall and Young's single-step max-T), which holds the chance of any false
      flag at ``alpha`` without Bonferroni's slack. The standard errors are recomputed on every shuffle, so their
      noise is in the reference distribution. A shuffle that leaves a tested cluster with a zero standard error
      counts as an infinite Q and ``|z|``, which can only make the test conservative. Plugging in the estimated
      common effect makes the test approximate rather than exact, and the shuffle requires that a respondent's
      rows share one arm.
    - ``"auto"`` (default): permutation when any tested cluster has fewer than ``CHI2_MIN_ARM`` (100) units in its
      smaller arm, else chi-square. ``attrs["method"]`` says which ran.

    A cluster is tested when it has at least ``min_respondents`` units, both arms, and a positive standard error. A
    cluster with a standard error of zero (``zero_variance``: within each arm, every unit's mean outcome equals the arm
    mean, as when an interviewer records the same answers for everyone) is left out of the test and reported, because
    that is itself a finding. ``flagged`` marks tested clusters with ``p_adjusted`` below ``alpha``.

    Args:
        df: One row per outcome observation.
        outcome: Numeric outcome column.
        treatment: 0/1 or boolean treatment column.
        cluster: Cluster column, e.g. the interviewer.
        respondent: Respondent column when a respondent contributes several rows.
        min_respondents: Smallest cluster, in units, that enters the test.
        alpha: Family-wise error rate for ``flagged``.
        method: ``"auto"``, ``"chi2"`` or ``"permutation"``, as above.
        n_permutations: Shuffles for the permutation reference; the smallest attainable p-value is
            ``1 / (n_permutations + 1)``.
        seed: Seed for the shuffles.

    Returns:
        One row per cluster with ``respondents``, ``treated``, ``control`` (units per arm), ``effect``, ``se``, ``z``,
        ``p_value``, ``p_adjusted``, ``zero_variance``, ``tested`` and ``flagged``. ``attrs`` holds the pooled
        ``effect``, its ``se`` clustered on respondents and ``se_cluster`` clustered on ``cluster``, the heterogeneity
        test (``q``, ``df``, ``p_value``, the inverse-variance weighted ``centre``, the ``method`` used,
        ``n_permutations`` (0 for chi-square) and ``min_arm``, the smallest arm among tested clusters) and
        ``zero_variance`` (the clusters with a zero standard error).
    """
    if method not in ("auto", "chi2", "permutation"):
        raise ValueError(f"method must be 'auto', 'chi2' or 'permutation', not {method!r}")
    y = df[outcome].to_numpy(dtype=np.float64)
    if np.isnan(y).any():
        raise ValueError(f"Column {outcome!r} has missing values; drop them first")
    treated = _binary(df[treatment], treatment)
    units = df[respondent].to_numpy() if respondent is not None else np.arange(len(df))
    codes, values = labels(df, cluster)
    pooled, pooled_se = _difference_in_means(y, treated, units)
    _, cluster_se = _difference_in_means(y, treated, codes)

    rows = []
    for code, value in enumerate(values):
        own = codes == code
        effect, se = _difference_in_means(y[own], treated[own], units[own])
        own_units = pd.DataFrame({"unit": units[own], "treated": treated[own]}).drop_duplicates("unit")
        rows.append(
            {
                cluster: value,
                "respondents": len(own_units),
                "treated": int(own_units["treated"].sum()),
                "control": int((~own_units["treated"]).sum()),
                "effect": effect,
                "se": se,
                "zero_variance": bool(se == 0),
            }
        )
    table = pd.DataFrame(rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        table["z"] = (table["effect"] - pooled) / table["se"]
    tested = ((table["respondents"] >= min_respondents) & np.isfinite(table["z"]) & (table["se"] > 0)).to_numpy()
    table["tested"] = tested

    effects, ses = table.loc[tested, "effect"].to_numpy(), table.loc[tested, "se"].to_numpy()
    q, centre = _q_statistic(effects, ses) if tested.any() else (np.nan, np.nan)
    q, centre = float(q), float(centre)
    dof = int(tested.sum() - 1)
    min_arm = int(table.loc[tested, ["treated", "control"]].min(axis=1).min()) if tested.any() else 0
    if method == "auto":
        method = "permutation" if tested.any() and min_arm < CHI2_MIN_ARM else "chi2"

    abs_z = np.abs(table.loc[tested, "z"].to_numpy())
    p_value = np.full(len(table), np.nan)
    p_adjusted = np.full(len(table), np.nan)
    if method == "permutation" and tested.any():
        rng = np.random.default_rng(seed)
        null_q, null_z = _permutation_null(y, treated, units, codes, centre, tested, n_permutations, rng)
        q_p = float(permutation_p(null_q[:, np.newaxis], np.array([q]))[0]) if dof > 0 else np.nan
        p_value[tested] = permutation_p(null_z, abs_z)
        p_adjusted[tested] = permutation_p(np.broadcast_to(null_z.max(axis=1, keepdims=True), null_z.shape), abs_z)
    else:
        q_p = float(stats.chi2.sf(q, dof)) if dof > 0 else np.nan
        p_value[tested] = 2 * stats.norm.sf(abs_z)
        p_adjusted = bonferroni(p_value)
        n_permutations = 0
    table["p_value"] = p_value
    table["p_adjusted"] = p_adjusted
    table["flagged"] = table["tested"] & (table["p_adjusted"] < alpha)
    table = table[
        [cluster, "respondents", "treated", "control", "effect", "se", "z", "p_value", "p_adjusted"]
        + ["zero_variance", "tested", "flagged"]
    ]
    table.attrs.update(
        effect=pooled,
        se=pooled_se,
        se_cluster=cluster_se,
        centre=centre,
        q=q,
        df=dof,
        p_value=q_p,
        method=method,
        n_permutations=n_permutations,
        min_arm=min_arm,
        zero_variance=table.loc[table["zero_variance"], cluster].tolist(),
    )
    return table


def assignment_balance(
    df: pd.DataFrame, treatment: Hashable, cluster: Hashable, *, alpha: float = 0.05
) -> pd.DataFrame:
    """Whether each cluster's share of treated units matches the overall share.

    Mechanism: under randomization at the unit level, the number of treated units in a cluster of size n is
    binomial with n draws and the sample-wide treated share. Each cluster gets an exact two-sided binomial test of its
    treated count against that share; ``flagged`` marks clusters below ``alpha`` after a Bonferroni correction.
    ``attrs`` adds a chi-square test of independence between cluster and treatment across all clusters. An
    interviewer who fills in interviews rather than conducting them may also skip the randomization, so a lopsided
    split is a process signal even when the answers look ordinary.

    Pass one row per unit of randomization (e.g. one per respondent, not one per answer).

    Args:
        df: One row per randomized unit.
        treatment: 0/1 or boolean treatment column.
        cluster: Cluster column.
        alpha: Family-wise error rate for ``flagged``.

    Returns:
        One row per cluster with ``units``, ``treated``, ``share_treated``, ``expected_treated``, ``p_value``,
        ``p_adjusted`` and ``flagged``. ``attrs`` holds the overall ``share_treated`` and the ``chi2``, ``df`` and
        ``p_value`` of the independence test.
    """
    treated = _binary(df[treatment], treatment)
    codes, values = labels(df, cluster)
    overall = treated.mean()
    units = np.bincount(codes, minlength=len(values))
    n_treated = np.bincount(codes, weights=treated, minlength=len(values)).astype(np.int64)
    p_values = np.array([stats.binomtest(int(k), int(n), overall).pvalue for k, n in zip(n_treated, units)])
    table = pd.DataFrame(
        {
            cluster: values,
            "units": units,
            "treated": n_treated,
            "share_treated": n_treated / units,
            "expected_treated": units * overall,
            "p_value": p_values,
        }
    )
    table["p_adjusted"] = bonferroni(p_values)
    table["flagged"] = table["p_adjusted"] < alpha
    crosstab = np.column_stack([n_treated, units - n_treated])
    chi2, p_value, dof, _ = stats.chi2_contingency(crosstab, correction=False)
    table.attrs.update(share_treated=float(overall), chi2=float(chi2), df=int(dof), p_value=float(p_value))
    return table
