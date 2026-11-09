"""Checks on an estimate and on a randomized treatment, cluster by cluster."""

from __future__ import annotations

from collections.abc import Callable, Hashable

import numpy as np
import pandas as pd
from scipy import stats

from interviewer_checks._util import bonferroni, labels


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


def effect_heterogeneity(
    df: pd.DataFrame,
    outcome: Hashable,
    treatment: Hashable,
    cluster: Hashable,
    *,
    respondent: Hashable | None = None,
    min_respondents: int = 20,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Each cluster's own treatment effect, compared with the pooled effect.

    Mechanism: within each cluster (usually an interviewer), the effect is the difference between the mean outcome
    of treated and control rows. Its standard error is clustered on ``respondent`` when the data are long (several
    outcome rows per respondent), otherwise each row is its own unit. ``z`` is ``(effect - pooled) / se``: how many of
    its own standard errors a cluster's effect sits from the effect in the whole sample.

    The heterogeneity test is Cochran's Q, the standard test from meta-analysis: weight each tested cluster's effect
    by its inverse variance, take the weighted mean, and sum the weighted squared deviations from it. If treatment is
    randomized within clusters and clusters differ only by sampling noise, Q is approximately chi-square with one
    fewer degree of freedom than tested clusters. (Summing the squared ``z`` against the pooled difference in means
    gives a slightly larger number, because the weighted mean is the centre that minimizes the sum.)

    A cluster is tested when it has at least ``min_respondents`` units, both arms, and a positive standard error. A
    cluster with a standard error of zero (``zero_variance``: within each arm, every unit's mean outcome equals the arm
    mean, as when an interviewer records the same answers for everyone) is left out of the test and reported, because
    that is itself a finding. ``flagged`` marks tested clusters whose two-sided p-value for ``z`` stays below
    ``alpha`` after a Bonferroni correction for the number of tested clusters.

    Args:
        df: One row per outcome observation.
        outcome: Numeric outcome column.
        treatment: 0/1 or boolean treatment column.
        cluster: Cluster column, e.g. the interviewer.
        respondent: Respondent column when a respondent contributes several rows.
        min_respondents: Smallest cluster, in units, that enters the test.
        alpha: Family-wise error rate for ``flagged``.

    Returns:
        One row per cluster with ``respondents``, ``treated``, ``control`` (units per arm), ``effect``, ``se``, ``z``,
        ``p_value``, ``p_adjusted``, ``zero_variance``, ``tested`` and ``flagged``. ``attrs`` holds the pooled
        ``effect``, its ``se`` clustered on respondents and ``se_cluster`` clustered on ``cluster``, and the
        heterogeneity test (``chi2``, which is Cochran's Q, ``df``, ``p_value``, and the inverse-variance weighted
        ``centre``) plus ``zero_variance`` (the clusters with a zero standard error).
    """
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
    table["tested"] = (table["respondents"] >= min_respondents) & np.isfinite(table["z"]) & (table["se"] > 0)
    table["p_value"] = np.where(table["tested"], 2 * stats.norm.sf(np.abs(table["z"])), np.nan)
    table["p_adjusted"] = bonferroni(table["p_value"].to_numpy())
    table["flagged"] = table["tested"] & (table["p_adjusted"] < alpha)
    table = table[
        [cluster, "respondents", "treated", "control", "effect", "se", "z", "p_value", "p_adjusted"]
        + ["zero_variance", "tested", "flagged"]
    ]

    tested = table["tested"].to_numpy()
    weights = 1.0 / table.loc[tested, "se"].to_numpy() ** 2
    effects = table.loc[tested, "effect"].to_numpy()
    centre = float(np.sum(weights * effects) / np.sum(weights)) if tested.any() else np.nan
    chi2 = float(np.sum(weights * (effects - centre) ** 2))
    dof = int(tested.sum() - 1)
    table.attrs.update(
        effect=pooled,
        se=pooled_se,
        se_cluster=cluster_se,
        centre=centre,
        chi2=chi2,
        df=dof,
        p_value=float(stats.chi2.sf(chi2, dof)) if dof > 0 else np.nan,
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
