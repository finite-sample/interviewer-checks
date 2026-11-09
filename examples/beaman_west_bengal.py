"""Case study: Beaman, Duflo, Pande and Topalova's (2012) West Bengal reservation experiment, adolescent survey.

Usage: uv run python examples/beaman_west_bengal.py path/to/beaman/data

The directory must hold the J-PAL "Powerful Women in India" files ``powerful_women_in_india_teenager_survey.dta``,
``..._pradhan_seats_reserved_for_women.dta`` and ``..._household_survey.dta`` (Harvard Dataverse,
doi:10.7910/DVN/O3UKFO). The data are not distributed with this package.
"""

import itertools
import sys

import numpy as np
import pandas as pd

from interviewer_checks import (
    assignment_balance,
    cluster_influence,
    effect_heterogeneity,
    interview_clock,
    interviewer_similarity,
    percent_match,
)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
pd.set_option("display.precision", 3)

FILE = "{}/powerful_women_in_india_{}.dta"
HOUSEWIFE = {
    "house wife",
    "housewife",
    "house hold work",
    "household work",
    "house work",
    "what in-laws want",
    "what in laws want",
    "what husband wants",
    "what husband & in-laws want",
    "what in-laws & husband want",
    "what in-laws & husband  want",
    "what in-laws & husbnand want",
    "what father-in-law wants",
    "what parents want",
    "what parent's want",
    "parents know",
    "parent's know",
    "perents know",
    "prarents know",
}
# Occupations that need an education. Counting nurses as well lifts the never-reserved girls' share from 0.16 to
# 0.21 against the paper's 0.166, so the paper evidently did not count them.
EDUCATED_JOB = r"teach|taecher|doctor|engineer|professor|lawyer|iawyer|i\.p\.s|officer|manager"
OUTCOMES = ["no_housewife", "educated_job", "marry_after_18", "graduate"]
# Table 2 of the paper: never-reserved means (boys, girls) and the twice-reserved change in the girl-boy gap (SE).
PAPER_MEANS = {
    "no_housewife": (0.998, 0.600),
    "educated_job": (0.157, 0.166),
    "marry_after_18": (0.980, 0.660),
    "graduate": (0.296, 0.195),
    "index": (0.248, -0.260),
}
PAPER_GAP_CHANGE = {
    "no_housewife": (0.075, 0.042),
    "educated_job": (0.099, 0.051),
    "marry_after_18": (0.085, 0.049),
    "graduate": (0.001, 0.045),
    "index": (0.166, 0.057),
}
YES_NO = [f"C1_{i}" for i in range(1, 18) if i != 15]
ITEMS = ["B1_1", "B1_2", "B1_4", *YES_NO, "D_1", "D_2", *[f"E_{i}" for i in range(1, 11)]]


def load(root: str) -> pd.DataFrame:
    """One row per adolescent with interviewer, geography, reservation arm, the four outcomes and the clock."""
    teens = pd.read_stata(FILE.format(root, "teenager_survey"), convert_categoricals=False)
    seats = pd.read_stata(FILE.format(root, "pradhan_seats_reserved_for_women"), convert_categoricals=False)
    seats["twice"] = (seats["res_woman"] == 1) & (seats["prev_res_woman"] == 1)
    seats["once"] = ((seats["res_woman"] == 1) | (seats["prev_res_woman"] == 1)) & ~seats["twice"]
    df = teens.merge(seats[["AA0_2b", "twice", "once"]], on="AA0_2b", how="inner", validate="many_to_one")
    df = df[df["gendercode"].isin([1, 2])].copy()

    df["interviewer"] = df["AA0_9_1"].astype(int)
    df["block"] = df["AA0_1b"].astype(int)
    df["gp"] = df["AA0_2b"].astype(int)
    df["village"] = df["gp"].astype(str) + "-" + df["AA0_3b"].astype(int).astype(str)
    df["girl"] = (df["gendercode"] == 2).astype(int)
    df["arm"] = np.select([df["twice"], df["once"]], ["twice", "once"], "never")

    occupation = df["B1_3"].str.strip().str.lower()
    df["no_housewife"] = (~occupation.isin(HOUSEWIFE)).astype(float)
    df["educated_job"] = occupation.str.contains(EDUCATED_JOB, regex=True).astype(float)
    marriage = df["B1_2"].where(df["B1_2"] < 777)
    df["marry_after_18"] = (marriage >= 19).astype(float).where(marriage.notna())
    df["graduate"] = (df["B1_1"] == 13).astype(float).where(df["B1_1"].notna())
    never = df["arm"] == "never"
    z = [(df[column] - df.loc[never, column].mean()) / df.loc[never, column].std() for column in OUTCOMES]
    df["index"] = pd.concat(z, axis=1).mean(axis=1)

    for column in ITEMS:
        df[column] = df[column].where(df[column] < 777)
    df["start"] = df["AA0_12_hour"] * 60 + df["AA0_12_min"]
    df["end"] = df["AA0_13_hour"] * 60 + df["AA0_13_min"]
    df["day"] = df["AA0_8_month"].astype(int) * 100 + df["AA0_8_day"].astype(int)
    return df.reset_index(drop=True)


def gap_change(df: pd.DataFrame, outcome: str = "index") -> float:
    """Change in the girl-minus-boy gap, twice-reserved against never-reserved councils.

    With a dummy for each arm, for girls and for each arm-by-girl cell, OLS reproduces the cell means, so the
    twice-by-girl coefficient is exactly this difference in differences of means.
    """
    means = df.groupby(["arm", "girl"])[outcome].mean()
    return float((means["twice", 1] - means["twice", 0]) - (means["never", 1] - means["never", 0]))


def replication(df: pd.DataFrame) -> pd.DataFrame:
    never = df[df["arm"] == "never"].groupby("girl")
    rows = []
    for outcome in [*OUTCOMES, "index"]:
        jackknife = cluster_influence(df.dropna(subset=[outcome]), lambda d, o=outcome: gap_change(d, o), "gp")
        rows.append(
            {
                "outcome": outcome,
                "boys_paper": PAPER_MEANS[outcome][0],
                "boys": never[outcome].mean()[0],
                "girls_paper": PAPER_MEANS[outcome][1],
                "girls": never[outcome].mean()[1],
                "change_paper": PAPER_GAP_CHANGE[outcome][0],
                "change": jackknife.attrs["estimate"],
                "se_paper": PAPER_GAP_CHANGE[outcome][1],
                "se_gp_jackknife": jackknife.attrs["jackknife_se"],
            }
        )
    return pd.DataFrame(rows)


def gap_cells(df: pd.DataFrame) -> pd.DataFrame:
    """One row per interviewer x council with both girls and boys: the girls' mean index minus the boys'.

    Within an interviewer, the twice-reserved cells' mean gap minus the never-reserved cells' mean gap is that
    interviewer's version of the headline difference in differences, so effect_heterogeneity can compare them.
    """
    use = df[df["arm"].isin(["twice", "never"])]
    means = use.groupby(["interviewer", "gp", "arm", "girl"])["index"].mean().unstack("girl").dropna()
    cells = (means[1] - means[0]).rename("gap").reset_index()
    cells["twice"] = (cells["arm"] == "twice").astype(int)
    return cells


def exact_p(cells: pd.DataFrame, pooled: float) -> float:
    """Exact two-sided p-value for "this interviewer's effect equals the pooled one".

    Subtract the pooled effect from the twice-reserved cells' gaps, so that under the null the arm labels carry no
    information, then enumerate every way of choosing which cells are twice-reserved. The p-value is the share of
    relabelings whose difference in means is at least as large in absolute value as the observed one. With three or
    four cells in an arm, the standard error behind effect_heterogeneity's z is itself noisy and the normal reference
    distribution overstates how surprising a large z is; the enumeration needs neither.
    """
    treated = cells["twice"].to_numpy().astype(bool)
    gaps = cells["gap"].to_numpy() - pooled * treated
    observed = gaps[treated].mean() - gaps[~treated].mean()
    effects = np.array(
        [
            gaps[list(chosen)].mean() - np.delete(gaps, list(chosen)).mean()
            for chosen in itertools.combinations(range(len(gaps)), int(treated.sum()))
        ]
    )
    return float(np.mean(np.abs(effects) >= abs(observed) - 1e-12))


def relabeled_q(cells: pd.DataFrame, observed: float, pooled: float, draws: int = 1999) -> float:
    """Permutation p-value for Cochran's Q: the share of relabelings with a Q at least as large as observed.

    As in :func:`exact_p`, the pooled effect is taken out of the twice-reserved gaps, and the arm labels are then
    shuffled within each interviewer, keeping how many cells each interviewer has in each arm. Every interviewer then
    has the same true effect, so this is the null distribution of Q with the same few-cell standard errors.
    """
    rng = np.random.default_rng(0)
    adjusted = cells.assign(gap=cells["gap"] - pooled * cells["twice"])
    labels = adjusted.groupby("interviewer")["twice"]
    null = []
    for _ in range(draws):
        shuffled = adjusted.assign(twice=labels.transform(lambda arm: rng.permutation(arm.to_numpy())))
        null.append(effect_heterogeneity(shuffled, "gap", "twice", "interviewer", min_respondents=4).attrs["chi2"])
    return float((1 + np.sum(np.array(null) >= observed - 1e-12)) / (1 + draws))


def main(root: str) -> None:
    df = load(root)
    per_interviewer = df.groupby("interviewer").size()
    per_village = df.groupby("village")["interviewer"].nunique()
    per_gp = df.groupby("gp")["interviewer"].nunique()
    print(
        f"{len(df):,} adolescents, {df['block'].nunique()} blocks, {df['gp'].nunique()} councils, "
        f"{df['village'].nunique()} villages, {len(per_interviewer)} interviewers "
        f"(median {per_interviewer.median():.0f} interviews; {(per_interviewer < 20).sum()} with fewer than 20)"
    )
    print(
        f"councils by arm: {df.groupby('arm')['gp'].nunique().to_dict()}; villages with one interviewer: "
        f"{(per_village == 1).mean():.0%}; councils with one interviewer: {(per_gp == 1).mean():.0%}\n"
    )

    print("## replication: never-reserved means and the twice-reserved change in the girl-boy gap (Table 2)")
    print(replication(df).to_string(index=False), "\n")

    print("## assignment_balance (one row per interviewer x council; twice-reserved against never-reserved)")
    pairs = df[df["arm"].isin(["twice", "never"])].drop_duplicates(["interviewer", "gp"])
    balance = assignment_balance(pairs.assign(twice=pairs["arm"].eq("twice").astype(int)), "twice", "interviewer")
    a = balance.attrs
    print(
        f"{a['share_treated']:.1%} of interviewer x council pairs twice-reserved; chi2 = {a['chi2']:.1f} on {a['df']} "
        f"df, p = {a['p_value']:.3f}; flagged {balance['flagged'].sum()}"
    )
    columns = ["interviewer", "units", "treated", "share_treated", "expected_treated", "p_value", "p_adjusted"]
    print(balance.sort_values("p_value").head(5)[columns].to_string(index=False))
    current = df.drop_duplicates(["interviewer", "gp"])
    current = current.assign(reserved=current["arm"].ne("never").astype(int))
    a = assignment_balance(current, "reserved", "interviewer").attrs
    print(
        f"ever reserved against never, all councils: {a['share_treated']:.1%}; chi2 = {a['chi2']:.1f} on {a['df']} "
        f"df, p = {a['p_value']:.3f}\n"
    )

    print("## effect_heterogeneity (girl-boy gap in the index per interviewer x council; twice vs never)")
    cells = gap_cells(df)
    arms = cells.groupby("interviewer")["twice"].agg(["sum", "size"])
    eligible = arms.index[(arms["sum"] >= 2) & (arms["size"] - arms["sum"] >= 2)]
    effects = effect_heterogeneity(
        cells[cells["interviewer"].isin(eligible)], "gap", "twice", "interviewer", min_respondents=4
    )
    a = effects.attrs
    print(
        f"{len(cells)} cells; {len(eligible)} interviewers with at least two cells in each arm. Pooled change "
        f"{a['effect']:.3f} (SE {a['se']:.3f}; {a['se_cluster']:.3f} clustered by interviewer); Cochran's Q = "
        f"{a['chi2']:.1f} on {a['df']} df, p = {a['p_value']:.3f}; flagged {effects['flagged'].sum()}"
    )
    columns = ["interviewer", "treated", "control", "effect", "se", "z", "p_adjusted"]
    print(effects.sort_values("z", key=np.abs, ascending=False).head(5)[columns].to_string(index=False))
    top = effects.loc[effects["z"].abs().idxmax(), "interviewer"]
    own = cells[cells["interviewer"] == top]
    p_exact = exact_p(own, effects.attrs["effect"])
    tested = len(eligible)
    print(f"exact relabeling test for {top}: p = {p_exact:.3f}; Bonferroni over {tested}: {p_exact * tested:.2f}")
    q_p = relabeled_q(cells[cells["interviewer"].isin(eligible)], a["chi2"], a["effect"])
    print(f"Cochran's Q against relabeling within interviewer: p = {q_p:.3f}\n")

    print("## cluster_influence (headline change in the gap, each interviewer left out)")
    influence = cluster_influence(df, gap_change, "interviewer")
    a = influence.attrs
    gp_se = cluster_influence(df, gap_change, "gp").attrs["jackknife_se"]
    print(
        f"full {a['estimate']:.3f}; interviewer-jackknife SE {a['jackknife_se']:.3f}; council-jackknife SE "
        f"{gp_se:.3f}; range without one interviewer {influence['estimate_without'].min():.3f} to "
        f"{influence['estimate_without'].max():.3f}"
    )
    columns = ["interviewer", "rows", "estimate_without", "shift", "shift_in_se"]
    print(influence.head(5)[columns].to_string(index=False), "\n")

    print(f"## percent_match within village ({len(ITEMS)} closed-ended items; chance from village marginals)")
    matches = percent_match(df, ITEMS, area="village", interviewer="interviewer", within_area=True)
    a = matches.attrs
    print(
        f"best match >= 0.85: {a['share_at_85']:.1%}; exact-duplicate pairs: {a['exact_duplicate_pairs']}; "
        f"beyond chance (independent items): {a['flagged']}"
    )
    flagged = df.loc[matches["flagged"]]
    print(f"flagged interviews by interviewer: {flagged['interviewer'].value_counts().to_dict()}\n")

    for area in ("village", "gp"):
        print(f"## interviewer_similarity (area = {area})")
        similarity = interviewer_similarity(df, ITEMS, "interviewer", area, n_permutations=9999, seed=1)
        a = similarity.attrs
        print(
            f"pooled excess {a['excess']:+.4f} (p = {a['p_value']:.4f}); {a['single_interviewer_interviews']:.0%} of "
            f"interviews in single-interviewer {area}s are not tested; flagged {similarity['flagged'].sum()}"
        )
        columns = ["interviewer", "tested", "nearest_own", "nearest_other", "excess", "p_adjusted"]
        top = similarity.sort_values(["p_value", "excess"], ascending=[True, False]).head(5)
        print(top[columns].to_string(index=False), "\n")

    print("## interview_clock (same interviewer and day; 5 minutes or less is short; over 4 hours untimed)")
    # Times are whole minutes, so a floor of 6 marks interviews logged at 5 minutes or less.
    times = interview_clock(
        df, "interviewer", "start", end="end", date="day", min_minutes=6, max_minutes=240
    ).sort_values("overlap_share", ascending=False)
    a = times.attrs
    rounded = (df[["AA0_12_min", "AA0_13_min"]] % 5 == 0).mean()
    print(
        f"timed {a['timed_share']:.1%}; overlapping the previous interview that day: {a['overlap_share']:.1%}; "
        f"5 minutes or less: {a['short_share']:.1%}; flagged {times['flagged'].sum()}; start and end minutes on a "
        f"multiple of 5: {rounded.iloc[0]:.0%}, {rounded.iloc[1]:.0%}"
    )
    print(times[times["interviews"] >= 20].to_string(index=False))
    fast = times.loc[(times["interviews"] >= 20) & (times["short_share"] > 0.5), "interviewer"].tolist()
    without = df[~df["interviewer"].isin(fast)]
    print(
        f"interviewers with most interviews at 5 minutes or less: {fast}; headline change without them: "
        f"{gap_change(without):.3f} ({len(without):,} rows)"
    )


if __name__ == "__main__":
    main(sys.argv[1])
