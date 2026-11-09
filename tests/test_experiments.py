import numpy as np
import pandas as pd
import pytest
from synthetic import make_survey

from interviewer_checks import assignment_balance, cluster_influence, effect_heterogeneity

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
        effects = effect_heterogeneity(df, "support", "treated", "interviewer")
        effect_p.extend(effects.loc[effects["tested"], "p_value"])
        heterogeneity_p.append(effects.attrs["p_value"])
        balance_p.extend(assignment_balance(df, "treated", "interviewer")["p_value"])
    assert 0.02 < np.mean(np.array(effect_p) < 0.05) < 0.09
    assert np.mean(np.array(balance_p) < 0.05) < 0.08
    assert np.mean(np.array(heterogeneity_p) < 0.05) <= 2 / 15


def test_planted_fabrication_moves_interviewer_effects():
    df = pakistan_style()
    table = effect_heterogeneity(df, "support", "treated", "interviewer").set_index("interviewer")
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
