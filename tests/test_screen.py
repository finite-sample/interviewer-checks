from synthetic import BATTERY, make_survey, questionnaire_items

from interviewer_checks import Screen, screen

FABRICATORS = [3, 17]
COPIERS = [5, 22]


def run(df):
    return screen(
        df,
        "interviewer",
        items=questionnaire_items(df),
        pattern_items=BATTERY,
        area="area",
        treatment="treated",
        outcome="support",
        n_permutations=1999,
    )


def test_clean_survey_raises_few_flags():
    result = run(make_survey(seed=8, areas=20))
    assert isinstance(result, Screen)
    assert result.table["n_flags"].sum() <= 2
    assert (result.table["n_flags"] < 2).all()
    assert "Flagged by two or more checks: none." in result.summary


def test_planted_fabrication_and_copying_rise_to_the_top():
    df = make_survey(
        seed=4, areas=20, interviews_per_interviewer=160, battery_level=1.5, fabricators=FABRICATORS, copiers=COPIERS
    )
    result = run(df)
    table = result.table.set_index("interviewer")
    for who in FABRICATORS:
        assert {"effect", "repetition"} <= set(table.loc[who, "flags"].split(", "))
    for who in COPIERS:
        assert "similarity" in table.loc[who, "flags"]
    assert table.drop(FABRICATORS + COPIERS)["n_flags"].max() <= 1
    assert set(result.table.head(2)["interviewer"]) == set(FABRICATORS)
    assert set(result.results) == {
        "effect_heterogeneity",
        "cluster_influence",
        "assignment_balance",
        "pattern_repetition",
        "straightlining",
        "percent_match",
        "interviewer_similarity",
    }


def test_screen_runs_only_the_checks_the_data_allow():
    df = make_survey(seed=9, areas=5)
    result = screen(df, "interviewer", pattern_items=BATTERY, n_permutations=99)
    assert set(result.results) == {"pattern_repetition", "straightlining"}
    assert "flag_effect" not in result.table and str(result) == result.summary
