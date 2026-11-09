import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from synthetic import BATTERY, make_survey, questionnaire_items

from interviewer_checks import interviewer_similarity, pattern_repetition, percent_match, straightlining
from interviewer_checks._util import hypergeometric_sum_sf, poisson_binomial_sf

FABRICATORS = [3, 17]
COPIERS = [5, 22]


def test_poisson_binomial_tail_matches_enumeration():
    probabilities = np.array([0.1, 0.5, 0.7, 0.3])
    for observed in range(6):
        brute = sum(
            np.prod([p if hit else 1 - p for p, hit in zip(probabilities, outcome)])
            for outcome in itertools.product([0, 1], repeat=4)
            if sum(outcome) >= observed
        )
        assert poisson_binomial_sf(probabilities[np.newaxis, :], [observed])[0] == pytest.approx(brute)


def test_hypergeometric_sum_matches_single_area_and_convolution():
    assert hypergeometric_sum_sf([(30, 6, 10)], 3) == pytest.approx(stats.hypergeom.sf(2, 30, 6, 10))
    first, second = stats.hypergeom.pmf(np.arange(5), 20, 4, 4), stats.hypergeom.pmf(np.arange(6), 25, 10, 5)
    joint = np.convolve(first, second)
    assert hypergeometric_sum_sf([(20, 4, 4), (25, 10, 5)], 4) == pytest.approx(joint[4:].sum())


def test_percent_match_agrees_with_a_pairwise_loop():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(0, 3, size=(40, 6)).astype(float), columns=list("abcdef"))
    df[df > 1.5] = np.nan
    table = percent_match(df, list("abcdef"), min_jointly_answered=0.5, chunk_size=7)
    values = df.to_numpy()
    for i in range(len(df)):
        shares = []
        for j in range(len(df)):
            both = ~np.isnan(values[i]) & ~np.isnan(values[j])
            if i != j and both.sum() >= 3:
                shares.append((values[i][both] == values[j][both]).mean())
        assert table["best_match"].iloc[i] == pytest.approx(max(shares) if shares else np.nan, nan_ok=True)


def test_percent_match_finds_a_planted_twin():
    df = make_survey(seed=0, areas=5)
    items = questionnaire_items(df)
    df.loc[7, items] = df.loc[123, items].to_numpy()
    table = percent_match(df, items, area="area", interviewer="interviewer")
    assert table.loc[7, "best_match"] == 1.0 and table.loc[7, "best_partner"] == 123
    assert table.loc[7, "exact_duplicates"] == 1 and table.attrs["exact_duplicate_pairs"] == 1
    assert table.loc[7, "flagged"] and table.loc[123, "flagged"]
    assert table["flagged"].sum() <= 4


def test_chunking_does_not_change_percent_match():
    df = make_survey(seed=1, areas=4)
    items = questionnaire_items(df)
    whole = percent_match(df, items, area="area", within_area=True)
    pieces = percent_match(df, items, area="area", within_area=True, chunk_size=13)
    pd.testing.assert_frame_equal(whole, pieces)


def test_expected_agreement_removes_spurious_matches_from_few_homogeneous_items():
    rng = np.random.default_rng(3)
    df = pd.DataFrame((rng.random((2000, 8)) < 0.9).astype(float), columns=[f"x{i}" for i in range(8)])
    df["area"] = rng.integers(0, 20, len(df))
    table = percent_match(df, [f"x{i}" for i in range(8)], area="area")
    assert table.attrs["share_at_85"] > 0.95
    assert table["expected_match"].mean() == pytest.approx(0.82, abs=0.02)
    assert table.attrs["flagged"] == 0


def test_percent_match_is_calibrated_when_items_are_independent():
    df = make_survey(seed=2, areas=15, loading=0.0)
    assert percent_match(df, questionnaire_items(df), area="area").attrs["flagged"] == 0


def test_percent_match_flags_copied_interviews():
    df = make_survey(seed=5, areas=15, copiers=COPIERS)
    table = percent_match(df, questionnaire_items(df), area="area", interviewer="interviewer")
    flagged = table[table["flagged"]]
    copied = df.loc[flagged.index, "interviewer"].isin(COPIERS)
    assert copied.sum() >= 20
    assert copied.mean() > 0.6
    assert flagged.loc[copied[copied].index, "partner_same_interviewer"].all()


def test_interviewer_similarity_flags_copiers_only():
    df = make_survey(seed=5, areas=15, copiers=COPIERS)
    table = interviewer_similarity(df, questionnaire_items(df), "interviewer", "area", n_permutations=1999, seed=0)
    table = table.set_index("interviewer")
    assert table.loc[COPIERS, "flagged"].all()
    assert table.drop(COPIERS)["flagged"].sum() == 0
    assert (table.loc[COPIERS, "excess"] > 0.03).all()


def test_pattern_repetition_catches_arm_specific_strings():
    df = make_survey(seed=4, areas=20, interviews_per_interviewer=80, fabricators=FABRICATORS)
    table = pattern_repetition(df, BATTERY, "interviewer", "area", strata="treated", n_simulations=1999, seed=0)
    table = table.set_index("interviewer")
    assert table.loc[FABRICATORS, "flagged"].all()
    assert table.drop(FABRICATORS)["flagged"].sum() == 0
    assert all("4-0-0-4" in text and "4-1-0-4" in text for text in table.loc[FABRICATORS, "modal_by_stratum"])


def test_straightliner_is_flagged_by_straightlining_not_by_repetition():
    df = make_survey(seed=6, areas=10)
    own = df.index[df["interviewer"] == 4]
    df.loc[own[:30], BATTERY] = 4.0
    repetition = pattern_repetition(df, BATTERY, "interviewer", "area", n_simulations=999, seed=0)
    flat = straightlining(df, BATTERY, "interviewer", "area").set_index("interviewer")
    assert flat.loc[4, "flagged"] and flat["flagged"].sum() == 1
    assert repetition.set_index("interviewer").loc[4, "p_value"] > 0.05


def test_single_interviewer_areas_are_reported_and_not_tested():
    df = make_survey(seed=7, areas=8, interviewers_per_area=1)
    for check in (
        straightlining(df, BATTERY, "interviewer", "area"),
        pattern_repetition(df, BATTERY, "interviewer", "area", n_simulations=99),
        interviewer_similarity(df, questionnaire_items(df), "interviewer", "area", n_permutations=99),
    ):
        assert check.attrs["single_interviewer_areas"] == 1.0
        assert check["p_value"].isna().all() and not check["flagged"].any()


def test_area_level_checks_are_calibrated_on_clean_surveys():
    flat, repeats, similar = [], [], []
    for seed in range(6):
        df = make_survey(seed=seed, areas=12, interviewers_per_area=3, interviews_per_interviewer=30, battery_level=1.5)
        flat.extend(straightlining(df, BATTERY, "interviewer", "area")["p_value"])
        repeats.extend(pattern_repetition(df, BATTERY, "interviewer", "area", n_simulations=499, seed=seed)["p_value"])
        similar.extend(
            interviewer_similarity(df, questionnaire_items(df), "interviewer", "area", n_permutations=499, seed=seed)[
                "p_value"
            ]
        )
    # Counts are discrete, so the tests are conservative; the second bound guards against p-values stuck at one.
    for p_values in (flat, repeats, similar):
        p_values = np.array(p_values)
        assert np.mean(p_values < 0.05) < 0.1
        assert np.mean(p_values < 0.5) > 0.2
