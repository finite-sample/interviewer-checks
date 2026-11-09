"""Checks on the answers themselves: near-duplicate interviews, repeated answer strings, straight-lining."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd

from interviewer_checks._util import (
    answer_strings,
    bonferroni,
    encode_items,
    hypergeometric_sum_sf,
    labels,
    numeric_values,
    one_hot,
    permutation_p,
    poisson_binomial_sf,
    shuffle_within,
    straightlined,
)


def _chance_agreement(
    indicators: np.ndarray,
    answered: np.ndarray,
    item_of_column: np.ndarray,
    area_codes: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Per-item probability that two interviews agree by chance, for each pair (first[i], second[i]).

    Same area: the share of all pairs of distinct interviews in that area who gave the same answer to the item,
    ``sum_k c_k (c_k - 1) / (n (n - 1))`` over the item's categories. This is the plug-in ``sum_k p_k**2`` with the
    small-sample bias removed, which matters in sampling points of a dozen interviews. Different areas:
    ``sum_k p_k(A) p_k(B)``, the chance that independent draws from the two areas agree. Items either interview
    left blank get probability zero, so they do not count.
    """
    n_areas = int(area_codes.max()) + 1
    counts = np.zeros((n_areas, indicators.shape[1]))
    np.add.at(counts, area_codes, indicators)
    answering = np.zeros((n_areas, answered.shape[1]))
    np.add.at(answering, area_codes, answered)
    starts = np.flatnonzero(np.r_[True, np.diff(item_of_column) != 0])

    area_a, area_b = area_codes[first], area_codes[second]
    n_a = answering[area_a]
    with np.errstate(divide="ignore", invalid="ignore"):
        same = np.add.reduceat(counts[area_a] * (counts[area_a] - 1), starts, axis=1) / (n_a * (n_a - 1))
        share_a = counts[area_a] / answering[area_a][:, item_of_column]
        share_b = counts[area_b] / answering[area_b][:, item_of_column]
        across = np.add.reduceat(np.nan_to_num(share_a * share_b), starts, axis=1)
    probability = np.where((area_a == area_b)[:, np.newaxis], same, across)
    jointly = answered[first] * answered[second]
    return np.nan_to_num(probability) * jointly


def percent_match(
    df: pd.DataFrame,
    items: Sequence[Hashable],
    *,
    area: Hashable | None = None,
    interviewer: Hashable | None = None,
    within_area: bool = False,
    min_jointly_answered: float = 0.5,
    alpha: float = 0.05,
    chunk_size: int = 2000,
) -> pd.DataFrame:
    """Each interview's closest other interview, and how much closer it is than chance.

    Mechanism, in two parts.

    1. Kuriakose and Robbins's percent match: for every pair of interviews, the share of items both answered on
       which they gave the same answer. ``best_match`` is an interview's highest share over all candidate partners
       (all other interviews, or only those in the same area with ``within_area=True``); a pair is a candidate only
       when both answered at least ``min_jointly_answered`` of the items. Missing answers are skipped; recode "don't
       know" to a value first if a copied interview would copy it.
    2. Expected agreement: a raw share means little on its own. Ten items on which 80% of people give the modal
       answer produce matches above 85% between strangers all the time, while ninety five-category items almost
       never do; and the more partners an interview has, the higher its best match is by luck alone. So for the best
       partner, each jointly answered item gets the probability that two interviews agree on it by chance, from the
       item's answer distribution in the pair's area (``area`` absorbs local homogeneity). ``expected_match`` is the
       mean of those probabilities and ``excess`` is ``best_match - expected_match``. ``p_value`` is the exact
       probability of at least the observed number of agreements if the items agreed independently with those
       probabilities, and ``p_adjusted`` multiplies it by the number of candidate pairs in the whole search
       (Bonferroni over every pair examined), which also pays for having picked the best of many partners.

    Item independence is the weak assumption: attitudes cluster, so genuine respondents agree more often than
    independent items would predict and ``p_value`` is too small for long correlated batteries. Treat ``flagged``
    as "worth looking at", and use :func:`interviewer_similarity`, whose null does not assume independence, to ask
    whether close matches concentrate within interviewers.

    Args:
        df: One row per interview.
        items: Columns compared.
        area: Area column (sampling point, district). Sets the answer distributions behind the expected agreement.
        interviewer: Interviewer column; adds whether the best partner has the same interviewer.
        within_area: Restrict partners to the same area.
        min_jointly_answered: Share of items both interviews must answer for the pair to count.
        alpha: Family-wise error rate for ``flagged``.
        chunk_size: Rows per block of the pairwise products; lower it if memory is short.

    Returns:
        One row per interview (same index as ``df``) with ``best_match``, ``best_partner`` (index label),
        ``jointly_answered``, ``candidates``, ``exact_duplicates`` (partners matching on every jointly answered item),
        ``expected_match``, ``excess``, ``p_value``, ``p_adjusted``, ``flagged``, and ``partner_same_area`` /
        ``partner_same_interviewer`` when those columns are given. ``attrs`` holds ``items``, ``candidate_pairs``,
        ``share_at_85`` (the share of interviews with a best match of at least 0.85, for comparison with the fixed
        threshold), ``exact_duplicate_pairs`` and ``flagged``.
    """
    if within_area and area is None:
        raise ValueError("within_area=True needs an area column")
    codes = encode_items(df, items)
    indicators, answered, item_of_column = one_hot(codes)
    area_codes, _ = labels(df, area)
    n_rows, n_items = codes.shape
    min_both = math.ceil(min_jointly_answered * n_items)

    best = np.full(n_rows, np.nan)
    partner = np.full(n_rows, -1, dtype=np.int64)
    jointly = np.zeros(n_rows, dtype=np.int64)
    agreed = np.zeros(n_rows, dtype=np.int64)
    candidates = np.zeros(n_rows, dtype=np.int64)
    exact = np.zeros(n_rows, dtype=np.int64)
    groups = [np.flatnonzero(area_codes == code) for code in np.unique(area_codes)] if within_area else [None]
    for members in groups:
        members = np.arange(n_rows) if members is None else members
        pool_indicators, pool_answered = indicators[members], answered[members]
        for start in range(0, len(members), chunk_size):
            rows = members[start : start + chunk_size]
            agree = indicators[rows] @ pool_indicators.T
            both = answered[rows] @ pool_answered.T
            eligible = both >= max(min_both, 1)
            eligible[np.arange(len(rows)), np.arange(start, start + len(rows))] = False
            share = np.where(eligible, agree / np.maximum(both, 1), -1.0)
            pick = share.argmax(axis=1)
            has = eligible.any(axis=1)
            index = np.arange(len(rows))
            best[rows] = np.where(has, share[index, pick], np.nan)
            partner[rows] = np.where(has, members[pick], -1)
            jointly[rows] = np.where(has, both[index, pick], 0)
            agreed[rows] = np.where(has, agree[index, pick], 0)
            candidates[rows] = eligible.sum(axis=1)
            exact[rows] = (eligible & (agree == both)).sum(axis=1)

    has_partner = partner >= 0
    first, second = np.flatnonzero(has_partner), partner[has_partner]
    probability = _chance_agreement(indicators, answered, item_of_column, area_codes, first, second)
    expected = np.full(n_rows, np.nan)
    p_value = np.full(n_rows, np.nan)
    expected[first] = probability.sum(axis=1) / jointly[first]
    p_value[first] = poisson_binomial_sf(probability, agreed[first])
    candidate_pairs = int(candidates.sum() // 2)
    p_adjusted = np.minimum(p_value * max(candidate_pairs, 1), 1.0)

    table = pd.DataFrame(
        {
            "best_match": best,
            "best_partner": pd.array([df.index[j] if j >= 0 else pd.NA for j in partner], dtype=object),
            "jointly_answered": jointly,
            "candidates": candidates,
            "exact_duplicates": exact,
            "expected_match": expected,
            "excess": best - expected,
            "p_value": p_value,
            "p_adjusted": p_adjusted,
            "flagged": p_adjusted < alpha,
        },
        index=df.index,
    )
    for column, name in ((area, "partner_same_area"), (interviewer, "partner_same_interviewer")):
        if column is not None:
            own = df[column].to_numpy()
            same = np.zeros(n_rows, dtype=object)
            same[:] = pd.NA
            same[first] = own[first] == own[second]
            table[name] = pd.array(same, dtype="boolean")
    table.attrs.update(
        items=n_items,
        candidate_pairs=candidate_pairs,
        share_at_85=float(np.nanmean(best >= 0.85)) if has_partner.any() else np.nan,
        exact_duplicate_pairs=int(exact.sum() // 2),
        flagged=int(table["flagged"].sum()),
    )
    return table


def _shared_areas(area_codes: np.ndarray, interviewer_codes: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Which interviews sit in an area worked by two or more interviewers, and how much of the data that is.

    Only those interviews can be compared with a neighbour; in a single-interviewer area, interviewer and area are
    the same thing and no within-area check can separate them.
    """
    per_area = pd.Series(interviewer_codes).groupby(area_codes).nunique()
    shared_areas = per_area.index[per_area.to_numpy() >= 2]
    shared = np.isin(area_codes, shared_areas)
    return shared, {
        "areas": int(len(per_area)),
        "single_interviewer_areas": float(np.mean(per_area.to_numpy() == 1)),
        "single_interviewer_interviews": float(1.0 - shared.mean()),
    }


def interviewer_similarity(
    df: pd.DataFrame,
    items: Sequence[Hashable],
    interviewer: Hashable,
    area: Hashable | None = None,
    *,
    n_permutations: int = 9999,
    min_jointly_answered: float = 0.5,
    alpha: float = 0.05,
    seed: int | None = None,
) -> pd.DataFrame:
    """Whether an interviewer's interviews have closer twins among their own work than chance allows.

    Mechanism: agreement between two interviews is the share of jointly answered items with the same answer (as in
    :func:`percent_match`). For each interview, ``nearest_own`` is its highest agreement with another interview by
    the same interviewer in the same area, and ``nearest_other`` its highest agreement with an interview by someone
    else there. An interviewer who copies or invents interviews leaves some of them with a near twin among their own
    work, which raises the mean of ``nearest_own`` over their interviews; a mean of all pairwise agreements would
    dilute a few copies among many honest pairs.

    The null shuffles interviewer labels among the interviews of each area, keeping how many interviews each
    interviewer did there, and recomputes each interviewer's mean ``nearest_own``. Because the shuffle stays inside the
    area, local opinion (which makes neighbours alike) and the number of own interviews (more candidates give a
    higher best match) are the same under the null as in the data. ``expected`` is the null mean, ``excess`` the
    observed minus expected, ``p_value`` the share of shuffles at least as large, and ``flagged`` applies a Bonferroni
    correction across tested interviewers.

    Interviewer and area are confounded when an area has a single interviewer: nothing can be shuffled, so those
    interviews are not compared, and ``attrs`` reports how much of the data that removes. If most areas have one
    interviewer, pass a coarser area (district rather than sampling point) and say so, because the comparison then
    also absorbs differences between the sampling points inside it. Cost grows with the square of area size; with
    ``area=None`` on thousands of interviews, lower ``n_permutations``.

    Args:
        df: One row per interview.
        items: Columns compared.
        interviewer: Interviewer column.
        area: Area column; ``None`` treats the sample as one area.
        n_permutations: Shuffles for the null; the smallest attainable p-value is ``1 / (n_permutations + 1)``.
        min_jointly_answered: Share of items both interviews must answer for the pair to count.
        alpha: Family-wise error rate for ``flagged``.
        seed: Seed for the shuffles.

    Returns:
        One row per interviewer with ``interviews``, ``tested`` (interviews in areas shared with other interviewers),
        ``nearest_own``, ``nearest_other``, ``expected``, ``excess``, ``p_value``, ``p_adjusted`` and ``flagged``.
        ``attrs`` holds the pooled ``excess`` and ``p_value`` (all interviews at once) and the confounding summary:
        ``areas``, ``single_interviewer_areas`` and ``single_interviewer_interviews`` (shares).
    """
    codes = encode_items(df, items)
    indicators, answered, _ = one_hot(codes)
    interviewer_codes, interviewer_values = labels(df, interviewer)
    area_codes, _ = labels(df, area)
    n_interviewers = len(interviewer_values)
    min_both = max(math.ceil(min_jointly_answered * codes.shape[1]), 1)
    shared, confounding = _shared_areas(area_codes, interviewer_codes)
    rng = np.random.default_rng(seed)

    blocks = []
    for code in np.unique(area_codes[shared]):
        members = np.flatnonzero(area_codes == code)
        agree = indicators[members] @ indicators[members].T
        both = answered[members] @ answered[members].T
        similarity = np.where(both >= min_both, agree / np.maximum(both, 1), -1.0)
        np.fill_diagonal(similarity, -1.0)
        blocks.append((members, similarity))

    def nearest(draws: np.ndarray, similarity: np.ndarray, own: bool) -> np.ndarray:
        """Each interview's best agreement with a same-label (or other-label) interview, NaN when there is none."""
        same = draws[:, :, np.newaxis] == draws[:, np.newaxis, :]
        best = np.where(same if own else ~same, similarity[np.newaxis], -1.0).max(axis=2)
        return np.where(best >= 0, best, np.nan)

    def per_interviewer(draws: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sums and counts of non-missing values by label, one row per draw."""
        offset = draws + n_interviewers * np.arange(draws.shape[0])[:, np.newaxis]
        valid = np.isfinite(values)
        size = draws.shape[0] * n_interviewers
        total = np.bincount(offset[valid], weights=values[valid], minlength=size).reshape(-1, n_interviewers)
        count = np.bincount(offset[valid], minlength=size).reshape(-1, n_interviewers)
        return total, count

    observed_sum = np.zeros(n_interviewers)
    observed_count = np.zeros(n_interviewers)
    other_sum = np.zeros(n_interviewers)
    other_count = np.zeros(n_interviewers)
    for members, similarity in blocks:
        own = interviewer_codes[members][np.newaxis, :]
        total, count = per_interviewer(own, nearest(own, similarity, True))
        observed_sum += total[0]
        observed_count += count[0]
        total, count = per_interviewer(own, nearest(own, similarity, False))
        other_sum += total[0]
        other_count += count[0]

    null_sum = np.zeros((n_permutations, n_interviewers))
    null_count = np.zeros((n_permutations, n_interviewers))
    for members, similarity in blocks:
        own = interviewer_codes[members]
        batch = max(1, min(n_permutations, 20_000_000 // len(members) ** 2))
        for start in range(0, n_permutations, batch):
            draws = min(batch, n_permutations - start)
            shuffled = shuffle_within(np.zeros(len(own)), own, draws, rng)
            total, count = per_interviewer(shuffled, nearest(shuffled, similarity, True))
            null_sum[start : start + draws] += total
            null_count[start : start + draws] += count

    with np.errstate(divide="ignore", invalid="ignore"):
        nearest_own = observed_sum / observed_count
        null = null_sum / null_count
        pooled = observed_sum.sum() / observed_count.sum()
        pooled_null = null_sum.sum(axis=1) / null_count.sum(axis=1)
    tested = np.zeros(n_interviewers, dtype=np.int64)
    np.add.at(tested, interviewer_codes[shared], 1)
    testable = observed_count > 0
    observed = np.where(testable, nearest_own, np.nan)
    has_null = bool(blocks) and n_permutations > 0
    p_value = permutation_p(null, observed) if has_null else np.full(n_interviewers, np.nan)
    with np.errstate(invalid="ignore"):
        expected = np.where(testable, null.sum(axis=0) / np.isfinite(null).sum(axis=0), np.nan) if has_null else np.nan
    table = pd.DataFrame(
        {
            interviewer: interviewer_values,
            "interviews": np.bincount(interviewer_codes, minlength=n_interviewers),
            "tested": tested,
            "nearest_own": observed,
            "nearest_other": np.where(other_count > 0, other_sum / np.maximum(other_count, 1), np.nan),
            "expected": expected,
            "excess": observed - expected,
            "p_value": p_value,
        }
    )
    table["p_adjusted"] = bonferroni(table["p_value"].to_numpy())
    table["flagged"] = table["p_adjusted"] < alpha
    table.attrs.update(
        excess=float(pooled - pooled_null.mean()) if has_null else np.nan,
        p_value=float(permutation_p(pooled_null[:, np.newaxis], np.array([pooled]))[0]) if has_null else np.nan,
        **confounding,
    )
    return table


def _max_repeats(keys: np.ndarray, strings: np.ndarray, groups: int, n_strings: int) -> np.ndarray:
    """For each draw (row) and group, how many members carry the group's most common string.

    ``keys`` gives each member's group in each draw; ``strings`` its string code, with negative codes never counted.
    Counting is one bincount over (draw, group, string).
    """
    draws = keys.shape[0]
    counted = strings >= 0
    flat = (np.arange(draws)[:, np.newaxis] * groups + keys[:, counted]) * n_strings + strings[counted]
    counts = np.bincount(flat.ravel(), minlength=draws * groups * n_strings)
    return counts.reshape(draws, groups, n_strings).max(axis=2)


def pattern_repetition(
    df: pd.DataFrame,
    items: Sequence[Hashable],
    interviewer: Hashable,
    area: Hashable | None = None,
    *,
    strata: Hashable | None = None,
    exclude_straightlines: bool = True,
    n_simulations: int = 9999,
    alpha: float = 0.05,
    seed: int | None = None,
) -> pd.DataFrame:
    """How often an interviewer records the same full answer string, against what their area would produce.

    Mechanism: each interview's answers to ``items`` form one string (e.g. ``4-0-0-4``). ``repeated`` is the number
    of an interviewer's interviews that carry their single most common string; with ``strata`` (e.g. treatment arm)
    the most common string is taken within each stratum and the counts are added, so an interviewer who writes one
    string for control and another for treatment is caught. The null shuffles interviewer labels within each area
    (and stratum), keeping how many interviews each interviewer did there, and recounts. Local opinion therefore sets
    the benchmark: an interviewer working where everyone gives the same answers is compared with neighbours who face
    the same people. ``expected_share`` is the mean shuffled share and ``p_value`` the share of shuffles with at least
    as many repeats. Interviews in areas with no other interviewer cannot be shuffled and are not tested (``tested``
    counts the rest).

    ``exclude_straightlines`` (default) stops strings with the same answer on every item from counting as repeats,
    because "top answer on everything" is often the most common genuine string; :func:`straightlining` covers
    those separately.

    Args:
        df: One row per interview.
        items: Columns that form the answer string.
        interviewer: Interviewer column.
        area: Area column; ``None`` treats the sample as one area.
        strata: Optional column (e.g. treatment arm) within which strings are compared and shuffled.
        exclude_straightlines: Never count all-same strings as repeats.
        n_simulations: Shuffles for the null; the smallest attainable p-value is ``1 / (n_simulations + 1)``.
        alpha: Family-wise error rate for ``flagged`` (Bonferroni across tested interviewers).
        seed: Seed for the shuffles.

    Returns:
        One row per interviewer with ``interviews``, ``tested``, ``modal_pattern`` and ``modal_share`` (over all
        their interviews), ``repeated``, ``repeated_share`` and ``expected_share`` (over tested interviews),
        ``p_value``, ``p_adjusted``, ``flagged``, and ``modal_by_stratum`` when ``strata`` is given. ``attrs``
        holds the confounding summary from :func:`interviewer_similarity`.
    """
    strings = answer_strings(df, items)
    string_codes = pd.factorize(strings)[0].astype(np.int64)
    if exclude_straightlines:
        string_codes[straightlined(encode_items(df, items), numeric_values(df, items))] = -1
    interviewer_codes, interviewer_values = labels(df, interviewer)
    area_codes, _ = labels(df, area)
    stratum_codes, stratum_values = labels(df, strata)
    n_interviewers, n_strata = len(interviewer_values), len(stratum_values)
    shared, confounding = _shared_areas(area_codes, interviewer_codes)
    rng = np.random.default_rng(seed)

    rows = np.flatnonzero(shared)
    cells = area_codes[rows] * n_strata + stratum_codes[rows]
    codes_tested = string_codes[rows]
    n_strings = max(int(string_codes.max()) + 1, 1)
    groups = n_interviewers * n_strata

    def repeats(keys: np.ndarray) -> np.ndarray:
        return (
            _max_repeats(keys * n_strata + stratum_codes[rows], codes_tested, groups, n_strings)
            .reshape(keys.shape[0], n_interviewers, n_strata)
            .sum(axis=2)
        )

    observed = repeats(interviewer_codes[rows][np.newaxis, :])[0]
    null = np.zeros((n_simulations, n_interviewers))
    batch = max(1, min(n_simulations, 20_000_000 // (groups * n_strings)))
    for start in range(0, n_simulations, batch):
        draws = min(batch, n_simulations - start)
        null[start : start + draws] = repeats(shuffle_within(cells, interviewer_codes[rows], draws, rng))

    tested = np.bincount(interviewer_codes[rows], minlength=n_interviewers)
    records = []
    for code, value in enumerate(interviewer_values):
        own = np.flatnonzero(interviewer_codes == code)
        modal = pd.Series(strings[own])[string_codes[own] >= 0].value_counts()
        record = {
            interviewer: value,
            "interviews": len(own),
            "tested": int(tested[code]),
            "modal_pattern": modal.index[0] if len(modal) else None,
            "modal_share": modal.iloc[0] / len(own) if len(modal) else 0.0,
        }
        if strata is not None:
            parts = []
            for s_code, s_value in enumerate(stratum_values):
                in_stratum = own[stratum_codes[own] == s_code]
                top = pd.Series(strings[in_stratum])[string_codes[in_stratum] >= 0].value_counts()
                if len(top):
                    parts.append(f"{s_value}: {top.index[0]} ({top.iloc[0] / len(in_stratum):.2f})")
            record["modal_by_stratum"] = "; ".join(parts)
        records.append(record)
    table = pd.DataFrame(records)
    testable = tested > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        table["repeated"] = np.where(testable, observed, np.nan)
        table["repeated_share"] = table["repeated"] / tested
        table["expected_share"] = np.where(testable, null.mean(axis=0) / tested, np.nan)
    table["p_value"] = np.where(testable, permutation_p(null, observed.astype(np.float64)), np.nan)
    table["p_adjusted"] = bonferroni(table["p_value"].to_numpy())
    table["flagged"] = table["p_adjusted"] < alpha
    order = [interviewer, "interviews", "tested", "modal_pattern", "modal_share", "repeated", "repeated_share"]
    order += ["expected_share", "p_value", "p_adjusted", "flagged"]
    order += ["modal_by_stratum"] if strata is not None else []
    table = table[order]
    table.attrs.update(confounding)
    return table


def straightlining(
    df: pd.DataFrame,
    items: Sequence[Hashable],
    interviewer: Hashable,
    area: Hashable | None = None,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Each interviewer's share of straight-lined interviews (the same answer on every item).

    Mechanism: an interview is straight-lined when it answered at least two items and gave all of them the same
    value (the same number for numeric items, the same label otherwise). The null is the same as in
    :func:`pattern_repetition`: shuffle interviewer labels within each area. Under that shuffle the number of an
    interviewer's interviews that are straight-lined is, area by area, hypergeometric (drawing their interviews from
    the area's interviews, of which a known number are straight-lined), so the exact ``p_value`` for "at least this
    many" comes from adding those counts across areas. Conditioning on each area's total avoids treating a
    neighbour's small-sample rate as known. ``expected_share`` is the null mean. Interviews in single-interviewer
    areas cannot be shuffled and are not tested.

    Args:
        df: One row per interview.
        items: Columns of one battery; straight-lining only means something across items that share a scale.
        interviewer: Interviewer column.
        area: Area column; ``None`` uses the whole sample as one area.
        alpha: Family-wise error rate for ``flagged`` (Bonferroni across tested interviewers).

    Returns:
        One row per interviewer with ``interviews``, ``straightlined``, ``share``, ``tested``, ``expected_share``,
        ``p_value``, ``p_adjusted`` and ``flagged``. ``attrs`` holds the overall ``share`` and the confounding summary.
    """
    flat = straightlined(encode_items(df, items), numeric_values(df, items))
    interviewer_codes, interviewer_values = labels(df, interviewer)
    area_codes, _ = labels(df, area)
    shared, confounding = _shared_areas(area_codes, interviewer_codes)
    area_size = np.bincount(area_codes)
    area_flat = np.bincount(area_codes, weights=flat).astype(np.int64)

    records = []
    for code, value in enumerate(interviewer_values):
        own = interviewer_codes == code
        tested = own & shared
        mine = np.bincount(area_codes[tested], minlength=len(area_size))
        cells = [(int(area_size[a]), int(area_flat[a]), int(mine[a])) for a in np.flatnonzero(mine)]
        observed = int(flat[tested].sum())
        records.append(
            {
                interviewer: value,
                "interviews": int(own.sum()),
                "straightlined": int(flat[own].sum()),
                "share": float(flat[own].mean()),
                "tested": int(tested.sum()),
                "expected_share": sum(k * n / big for big, k, n in cells) / tested.sum() if cells else np.nan,
                "p_value": hypergeometric_sum_sf(cells, observed) if cells else np.nan,
            }
        )
    table = pd.DataFrame(records)
    table["p_adjusted"] = bonferroni(table["p_value"].to_numpy())
    table["flagged"] = table["p_adjusted"] < alpha
    table.attrs.update(share=float(flat.mean()), **confounding)
    return table
