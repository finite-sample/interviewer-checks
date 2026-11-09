"""Case study: Blair, Fair, Malhotra and Shapiro's (2013) Pakistan endorsement experiment.

Usage: uv run python examples/blair_pakistan.py path/to/fms09-data-weighted-20june.dta

The replication file is not distributed with this package.
"""

import sys

import pandas as pd

from interviewer_checks import effect_heterogeneity, pattern_repetition, percent_match, screen

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
pd.set_option("display.precision", 3)

# Polio vaccination, FCR reform, Durand line, madrassa curriculum; suffix a is control, b-e the four endorsers.
POLICIES = {"polio": "q800", "fcr": "q810", "durand": "q830", "curriculum": "q840"}
GROUPS = ["a", "b", "c", "d", "e"]


def load(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Interviews in one arm (one row each), policy answers in long format, and the closed-ended questionnaire."""
    raw = pd.read_stata(path, convert_categoricals=False)
    raw = raw[raw["name"].notna()].copy()
    raw["name"] = raw["name"].astype(int)
    interviews = raw[["name", "spn", "d24"]].rename(columns={"name": "interviewer", "spn": "point", "d24": "district"})
    keep = pd.Series(True, index=raw.index)
    for policy, stem in POLICIES.items():
        # 98/99 are don't know / no response; a usable respondent has exactly one substantive answer per policy.
        block = raw[[stem + group for group in GROUPS]].where(lambda frame: frame.isin([1, 2, 3, 4, 5]))
        keep &= block.notna().sum(axis=1) == 1
        # 1 = "a great deal" ... 5 = "not at all"; flip to 0-4 so 4 is the most support.
        interviews[policy] = 5 - block.max(axis=1)
        interviews[policy + "_endorsed"] = block[stem + "a"].isna()
    interviews = interviews[keep]
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "respondent": interviews.index,
                    "interviewer": interviews["interviewer"],
                    "response": interviews[policy] / 4,
                    "endorsed": interviews[policy + "_endorsed"].astype(int),
                }
            )
            for policy in POLICIES
        ]
    )
    endorsed = interviews[[policy + "_endorsed" for policy in POLICIES]]
    # Respondents with some endorsed and some control items belong to neither arm.
    interviews = interviews[endorsed.all(axis=1) | ~endorsed.any(axis=1)].copy()
    interviews["treated"] = endorsed.loc[interviews.index].all(axis=1).astype(int)
    interviews["support"] = interviews[list(POLICIES)].mean(axis=1) / 4

    items = [c for c in raw.columns if c.startswith("q") and raw[c].notna().mean() >= 0.5 and raw[c].nunique() > 1]
    questionnaire = raw[["name", "spn", *items]].rename(columns={"name": "interviewer", "spn": "point"})
    return interviews, long, questionnaire


def main(path: str) -> None:
    interviews, long, questionnaire = load(path)
    print(f"{len(long) // 4:,} respondents, {interviews['interviewer'].nunique()} interviewers\n")

    print("## effect_heterogeneity (long format, SEs clustered on respondent)")
    effects = effect_heterogeneity(long, "response", "endorsed", "interviewer", respondent="respondent")
    a = effects.attrs
    print(
        f"pooled ATE {a['effect']:.3f} (SE {a['se']:.3f}); Cochran's Q = {a['chi2']:.1f} on {a['df']} df, "
        f"p = {a['p_value']:.4f}; zero standard error: {a['zero_variance']}"
    )
    columns = ["interviewer", "respondents", "treated", "control", "effect", "se", "z", "p_adjusted", "flagged"]
    print(effects.sort_values("z", ascending=False).head(5)[columns].to_string(index=False))
    rest = effect_heterogeneity(
        long[~long["interviewer"].isin([7, 37, 71])], "response", "endorsed", "interviewer", respondent="respondent"
    ).attrs
    print(
        f"without 7, 37, 71: ATE {rest['effect']:.3f}; Q = {rest['chi2']:.1f} on {rest['df']} df, "
        f"p = {rest['p_value']:.3f}\n"
    )

    print("## pattern_repetition (strings over the four policies, by arm, shuffled within district)")
    strings = pattern_repetition(interviews, list(POLICIES), "interviewer", "district", strata="treated", seed=0)
    strings["excess"] = strings["repeated_share"] - strings["expected_share"]
    columns = ["interviewer", "tested", "repeated_share", "expected_share", "p_adjusted", "modal_by_stratum"]
    print(strings.sort_values("excess", ascending=False).head(8)[columns].to_string(index=False))
    print(f"flagged: {strings.loc[strings['flagged'], 'interviewer'].tolist()}\n")

    print("## screen (one row per interview; outcome = mean support over the four policies)")
    result = screen(
        interviews,
        "interviewer",
        pattern_items=list(POLICIES),
        area="district",
        treatment="treated",
        outcome="support",
        seed=0,
    )
    print(result.summary, "\n")
    shown = ["interviewer", "interviews", "effect_z", "ate_shift", "share_treated", "repeated_share", "flags"]
    print(result.table.head(8)[shown].to_string(index=False), "\n")

    print("## percent_match over the whole questionnaire (chance agreement from sampling-point marginals)")
    items = [c for c in questionnaire.columns if c.startswith("q")]
    matches = percent_match(questionnaire, items, area="point", interviewer="interviewer")
    m = matches.attrs
    print(
        f"{m['items']} items; best match >= 0.85: {m['share_at_85']:.1%}; exact-duplicate pairs: "
        f"{m['exact_duplicate_pairs']}; beyond chance (independent items): {m['flagged']}"
    )
    within = percent_match(questionnaire, items, area="point", within_area=True).attrs
    print(
        f"same search within sampling points: best match >= 0.85: {within['share_at_85']:.1%}; "
        f"exact-duplicate pairs: {within['exact_duplicate_pairs']}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
