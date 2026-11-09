import numpy as np
import pandas as pd
import pytest
from scipy import stats
from synthetic import make_survey

import interviewer_checks.experiments as experiments
from interviewer_checks import assignment_balance, cluster_influence, effect_heterogeneity
from interviewer_checks._util import shuffle_within

FABRICATORS = [3, 17]


def pakistan_style(seed: int = 4) -> pd.DataFrame:
    return make_survey(seed=seed, areas=20, interviews_per_interviewer=160, battery_level=1.5, fabricators=FABRICATORS)


def test_cluster_influence_matches_a_hand_jackknife():
    df = pd.DataFrame({"g": [1, 1, 2, 2, 3, 3], "y": [0.0, 1.0, 2.0, 3.0, 10.0, 12.0]})
    table = cluster_influence(df, lambda frame: frame["y"].mean(), "g").set_index("g")
    leave_out = np.array([(2 + 3 + 10 + 12) / 4, (0 + 1 + 10 + 12) / 4, (0 + 1 + 2 + 3) / 4])
    assert table.loc[[1, 2, 3], "estimate_without"].to_numpy() == pytest.approx(leave_out)
    assert table.attrs["jackknife_se"] == pytest.approx(np.sqrt(2 / 3 * np.sum((leave_out - leave_out.mean()) ** 2)))
    assert table.index[0] == 3 and table.loc[3, "rank"] == 1


def test_row_level_se_is_the_textbook_formula():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=400), "t": rng.integers(0, 2, 400), "g": rng.integers(0, 4, 400)})
    attrs = effect_heterogeneity(df, "y", "t", "g", min_respondents=1).attrs
    treated, control = df.loc[df.t == 1, "y"], df.loc[df.t == 0, "y"]
    assert attrs["effect"] == pytest.approx(treated.mean() - control.mean())
    assert attrs["se"] == pytest.approx(
        np.sqrt(treated.var(ddof=0) / len(treated) + control.var(ddof=0) / len(control))
    )


def test_long_format_with_respondent_clusters_equals_respondent_means():
    rng = np.random.default_rng(1)
    respondents = 300
    wide = pd.DataFrame(rng.normal(size=(respondents, 4)), columns=list("abcd"))
    wide["t"] = rng.integers(0, 2, respondents)
    wide["g"] = rng.integers(0, 5, respondents)
    wide["id"] = np.arange(respondents)
    wide["mean"] = wide[list("abcd")].mean(axis=1)
    long = wide.melt(id_vars=["id", "t", "g"], value_vars=list("abcd"), value_name="y")
    from_long = effect_heterogeneity(long, "y", "t", "g", respondent="id", min_respondents=1)
    from_wide = effect_heterogeneity(wide, "mean", "t", "g", min_respondents=1)
    assert from_long.attrs["se"] == pytest.approx(from_wide.attrs["se"])
    assert from_long["z"].to_numpy() == pytest.approx(from_wide["z"].to_numpy())


def test_zero_variance_cluster_is_reported_and_left_out_of_the_test():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": rng.normal(size=300), "t": rng.integers(0, 2, 300), "g": np.repeat([0, 1, 2], 100)})
    df.loc[df.g == 2, "y"] = 3.0
    table = effect_heterogeneity(df, "y", "t", "g")
    assert table.attrs["zero_variance"] == [2]
    assert not table.set_index("g").loc[2, "tested"]
    assert table.attrs["df"] == 1


def test_rejects_non_binary_treatment():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "t": [0, 1, 2], "g": [1, 1, 1]})
    with pytest.raises(ValueError, match="0/1"):
        effect_heterogeneity(df, "y", "t", "g")


def test_clean_surveys_reject_at_the_nominal_rate():
    effect_p, balance_p, heterogeneity_p = [], [], []
    for seed in range(15):
        df = make_survey(seed=seed, areas=20)
        effects = effect_heterogeneity(df, "support", "treated", "interviewer", n_permutations=999, seed=seed)
        assert effects.attrs["method"] == "permutation"
        effect_p.extend(effects.loc[effects["tested"], "p_value"])
        heterogeneity_p.append(effects.attrs["p_value"])
        balance_p.extend(assignment_balance(df, "treated", "interviewer")["p_value"])
    assert 0.02 < np.mean(np.array(effect_p) < 0.05) < 0.09
    assert np.mean(np.array(balance_p) < 0.05) < 0.08
    assert np.mean(np.array(heterogeneity_p) < 0.05) <= 2 / 15


def test_planted_fabrication_moves_interviewer_effects():
    df = pakistan_style()
    table = effect_heterogeneity(df, "support", "treated", "interviewer", n_permutations=1999, seed=0)
    table = table.set_index("interviewer")
    assert table.loc[FABRICATORS, "flagged"].all()
    assert (table.loc[FABRICATORS, "z"] > 0).all()
    assert table.drop(FABRICATORS)["flagged"].sum() <= 1
    assert table.attrs["p_value"] < 0.001


def test_planted_fabrication_skews_assignment():
    df = pakistan_style()
    table = assignment_balance(df, "treated", "interviewer")
    # A 30-40% treated share over 160 interviews is not always significant after correcting for 60 interviewers
    # (it was not for Blair et al.'s three); the fabricators should still be the most lopsided.
    assert set(table.nsmallest(2, "p_value")["interviewer"]) == set(FABRICATORS)
    assert table.set_index("interviewer").loc[FABRICATORS, "flagged"].any()
    assert (table.set_index("interviewer").loc[FABRICATORS, "share_treated"] < 0.4).all()
    assert table.loc[~table["interviewer"].isin(FABRICATORS), "flagged"].sum() == 0


def test_planted_fabrication_is_among_the_most_influential_clusters():
    df = pakistan_style()

    def ate(frame: pd.DataFrame) -> float:
        treated = frame["treated"] == 1
        return frame.loc[treated, "support"].mean() - frame.loc[~treated, "support"].mean()

    table = cluster_influence(df, ate, "interviewer")
    assert set(FABRICATORS) <= set(table.head(4)["interviewer"])
    assert (table.set_index("interviewer").loc[FABRICATORS, "shift"] < 0).all()


def few_per_arm(rng: np.random.Generator, interviewers: int = 30, per_arm: int = 4) -> pd.DataFrame:
    """A homogeneous effect of 0.5 with interviewer-level baselines and a handful of units in each arm."""
    cluster = np.repeat(np.arange(interviewers), 2 * per_arm)
    treated = np.tile(np.r_[np.ones(per_arm), np.zeros(per_arm)], interviewers).astype(int)
    y = rng.normal(size=interviewers)[cluster] + 0.5 * treated + rng.normal(size=len(cluster))
    return pd.DataFrame({"g": cluster, "t": treated, "y": y})


def test_permutation_holds_its_size_where_chi2_does_not():
    # 30 interviewers with 4 units per arm and the same effect everywhere: the standard errors are so noisy that Q
    # against chi-square rejects in about 70% of runs (and Bonferroni-on-z flags someone in about 60%), while the
    # within-interviewer shuffle stays near 5% for both. 60 runs keep this to a few seconds.
    rng = np.random.default_rng(11)
    rejections = {"chi2": [], "permutation": []}
    any_flag = {"chi2": [], "permutation": []}
    for run in range(60):
        df = few_per_arm(rng)
        for method in rejections:
            table = effect_heterogeneity(
                df, "y", "t", "g", min_respondents=1, method=method, n_permutations=199, seed=run
            )
            rejections[method].append(table.attrs["p_value"] < 0.05)
            any_flag[method].append(table["flagged"].any())
    assert np.mean(rejections["chi2"]) > 0.4
    assert np.mean(rejections["permutation"]) <= 0.12
    assert np.mean(any_flag["chi2"]) > 0.3
    assert np.mean(any_flag["permutation"]) <= 0.12


def test_permutation_still_finds_planted_heterogeneity():
    df = few_per_arm(np.random.default_rng(12))
    df.loc[df["g"] < 3, "y"] += 3.0 * df.loc[df["g"] < 3, "t"]
    table = effect_heterogeneity(df, "y", "t", "g", min_respondents=1, method="permutation", seed=0)
    assert table.attrs["p_value"] < 0.01
    assert set(table.loc[table["flagged"], "g"]) <= {0, 1, 2}


def test_auto_picks_permutation_for_small_arms_and_chi2_for_large_ones():
    rng = np.random.default_rng(13)
    small = effect_heterogeneity(few_per_arm(rng, 10, 4), "y", "t", "g", min_respondents=1, n_permutations=99)
    assert small.attrs["method"] == "permutation" and small.attrs["min_arm"] == 4
    assert small.attrs["n_permutations"] == 99
    large = effect_heterogeneity(few_per_arm(rng, 5, experiments.CHI2_MIN_ARM), "y", "t", "g")
    assert large.attrs["method"] == "chi2" and large.attrs["n_permutations"] == 0
    assert large.attrs["p_value"] == pytest.approx(stats.chi2.sf(large.attrs["q"], 4))
    # One interviewer one unit short of the threshold sends the whole test to the permutation.
    df = few_per_arm(rng, 5, experiments.CHI2_MIN_ARM)
    df = df.drop(df.index[(df["g"] == 0) & (df["t"] == 1)][:1])
    assert effect_heterogeneity(df, "y", "t", "g", n_permutations=99).attrs["method"] == "permutation"


def test_shuffle_within_keeps_every_cell_s_values():
    cells = np.repeat([0, 1, 2], [3, 5, 4])
    values = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=bool)
    draws = shuffle_within(cells, values, 500, np.random.default_rng(0))
    for cell in range(3):
        members = cells == cell
        assert (draws[:, members].sum(axis=1) == values[members].sum()).all()
        assert len({tuple(row) for row in draws[:, members]}) > 1


def test_permutation_null_keeps_each_interviewer_s_arm_sizes(monkeypatch):
    df = few_per_arm(np.random.default_rng(14), 6, 3)
    df = df.drop(df.index[[0, 7, 8]]).reset_index(drop=True)
    expected = df.groupby("g")["t"].sum().to_numpy()
    seen = []

    def spy(cells, values, draws, rng):
        out = shuffle_within(cells, values, draws, rng)
        seen.append((cells, out))
        return out

    monkeypatch.setattr(experiments, "shuffle_within", spy)
    effect_heterogeneity(df, "y", "t", "g", min_respondents=1, method="permutation", n_permutations=200)
    cells, draws = seen[0]
    per_cluster = np.stack([draws[:, cells == g].sum(axis=1) for g in range(6)], axis=1)
    assert (per_cluster == expected).all()


def test_vectorized_statistics_match_the_table_on_the_observed_labels():
    rng = np.random.default_rng(15)
    respondents = 120
    wide = pd.DataFrame({"id": np.arange(respondents), "g": rng.integers(0, 4, respondents)})
    wide["t"] = rng.integers(0, 2, respondents)
    long = wide.loc[wide.index.repeat(rng.integers(1, 4, respondents))].reset_index(drop=True)
    long["y"] = rng.normal(size=len(long)) + long["t"]
    table = effect_heterogeneity(long, "y", "t", "g", respondent="id", min_respondents=1, method="chi2")
    units = long.groupby(["g", "id"], sort=True).agg(sums=("y", "sum"), counts=("y", "size"), t=("t", "first"))
    groups = units.index.get_level_values("g").to_numpy()
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    effects, se, pooled = experiments._cluster_effects(
        units["sums"].to_numpy(), units["counts"].to_numpy(float), starts, units["t"].to_numpy(bool)[np.newaxis]
    )
    assert effects[0] == pytest.approx(table["effect"].to_numpy())
    assert se[0] == pytest.approx(table["se"].to_numpy())
    assert pooled[0] == pytest.approx(table.attrs["effect"])
    assert experiments._q_statistic(effects[0], se[0])[0] == pytest.approx(table.attrs["q"])


def test_permutation_refuses_respondents_in_both_arms():
    df = pd.DataFrame({"id": [1, 1, 2, 2, 3, 4], "g": 0, "t": [0, 1, 0, 0, 1, 1], "y": [0.0, 1, 2, 3, 4, 6]})
    with pytest.raises(ValueError, match="varies within 1 respondent"):
        effect_heterogeneity(df, "y", "t", "g", respondent="id", min_respondents=1, method="permutation")
    assert effect_heterogeneity(df, "y", "t", "g", respondent="id", min_respondents=1, method="chi2").attrs["df"] == 0
