"""Case study: the 2013-14 NCAER REDS/SEPRI household listing (92,996 households in 193 Indian villages).

Usage: uv run python examples/reds.py path/to/reds/data

The directory must hold ``Sepri1/HH`` and ``Sepri2/HH`` with ``SECTION01.dta`` to ``SECTION03.dta``. The data are
not distributed with this package.
"""

import difflib
import sys

import numpy as np
import pandas as pd

from interviewer_checks import interviewer_similarity, pattern_repetition, percent_match, straightlining

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
pd.set_option("display.precision", 3)

KEY = ["state", "district", "village", "q1_1"]
TIMES = ["month_interview", "date_interview", "hour_start", "minute_start", "hour_end", "minute_end"]
# Closed-ended items answered by nearly every household: dwelling (section 2), food security (3), head (1).
HOUSING = ["q2_2", "q2_3", "q2_5", "q2_6", "q2_7a", "q2_7b", "q2_8a", "q2_8b", "q2_9", "q2_9a", "q2_10", "q2_12"]
HUNGER = [f"q3_2{i}" for i in range(1, 10)]  # nine never/rarely/sometimes/often items on going without food
FOOD = ["q3_1", *HUNGER, "q3_30", "q3_31"]
HEAD = ["q1_8", "q1_9", "q1_11", "q1_18", "q1_21", "q1_22"]
ITEMS = HOUSING + FOOD + HEAD
SEGMENT = 50  # households per stretch of the listing order
DRAWS = 49_999  # smallest p is 2e-5, which survives Bonferroni over ~600 interviewers


def canonical_names(names: pd.Series) -> pd.Series:
    """Map typed variants of an interviewer's name (KASHI NATH, KASHINATH, KAHSI NATH) to one spelling.

    Names were keyed in by hand, so one person appears under many spellings. Within a village, names with at least
    10 interviews and 2% of the village are anchors; any other name joins the anchor it most resembles if the
    difflib similarity ratio is at least 0.8, and is left unresolved (NaN) otherwise.
    """
    counts = names.value_counts()
    anchors = list(counts.index[counts >= max(10, 0.02 * len(names))])

    def resolve(name: str) -> str | None:
        if name in anchors:
            return name
        scores = {anchor: difflib.SequenceMatcher(None, name, anchor).ratio() for anchor in anchors}
        best = max(scores, key=scores.get, default=None)
        return best if best is not None and scores[best] >= 0.8 else None

    return names.map({name: resolve(name) for name in counts.index})


def load(root: str) -> pd.DataFrame:
    """One row per household with the items, the canonical interviewer, the village and the listing segment."""
    waves = []
    for wave in ("Sepri1", "Sepri2"):
        path = f"{root}/{wave}/HH/SECTION0"
        frame = pd.read_stata(
            path + "1.dta", columns=KEY + HEAD + TIMES + ["interviewer_name"], convert_categoricals=False
        )
        for section, items in (("2", HOUSING), ("3", FOOD)):
            frame = frame.merge(
                pd.read_stata(path + section + ".dta", columns=KEY + items, convert_categoricals=False),
                on=KEY,
                how="left",
            )
        frame["wave"] = wave
        waves.append(frame)
    df = pd.concat(waves, ignore_index=True)
    place = df[["state", "district", "village"]].astype(int).astype(str)
    df["village_id"] = df["wave"] + "-" + place["state"] + "-" + place["district"] + "-" + place["village"]
    df["typed_name"] = df["interviewer_name"].str.upper().str.replace(r"[^A-Z]", "", regex=True)
    df["canonical"] = df.groupby("village_id")["typed_name"].transform(canonical_names)
    df["interviewer"] = df["wave"] + "-" + place["state"] + "-" + df["canonical"]
    df = df[df["canonical"].notna()].sort_values(["village_id", "q1_1"]).reset_index(drop=True)
    # The listing ID runs along streets, and each interviewer took short runs of it, so a stretch of the listing is
    # the closest thing to "the same neighbourhood" the data offer.
    df["rank"] = df.groupby("village_id").cumcount()
    df["segment"] = df["village_id"] + "#" + (df["rank"] // SEGMENT).astype(str)
    for column in TIMES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["start"] = df["hour_start"] * 60 + df["minute_start"]
    df["end"] = df["hour_end"] * 60 + df["minute_end"]
    df["day"] = df["month_interview"] * 100 + df["date_interview"]
    return df


def twins_by_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Exact 30-item twins among pairs in a village, by listing distance and whether the interviewer is shared.

    If neighbours were simply alike, the twin rate would fall with distance in the same way for both kinds of
    pair. A same-interviewer excess at a fixed distance is what the interviewer adds.
    """
    strings = df.groupby(ITEMS, dropna=False).ngroup().to_numpy()
    rank, interviewer = df["rank"].to_numpy(), df["interviewer"].to_numpy()
    edges = np.array([1, 2, 3, 6, 11, 26, 51, 101])
    names = ["1", "2", "3-5", "6-10", "11-25", "26-50", "51-100", ">100"]
    pairs = np.zeros((2, len(names)))
    twins = np.zeros((2, len(names)))
    for _, members in df.groupby("village_id").indices.items():
        first, second = np.triu_indices(len(members), 1)
        first, second = members[first], members[second]
        band = np.digitize(rank[second] - rank[first], edges) - 1
        same = (interviewer[first] == interviewer[second]).astype(int)
        np.add.at(pairs, (same, band), 1)
        np.add.at(twins, (same, band), strings[first] == strings[second])
    table = pd.DataFrame(
        {
            "distance": names,
            "pairs_other": pairs[0].astype(int),
            "twins_other": twins[0] / pairs[0],
            "pairs_same": pairs[1].astype(int),
            "twins_same": twins[1] / pairs[1],
        }
    )
    table["ratio"] = table["twins_same"] / table["twins_other"]
    return table


def clock(df: pd.DataFrame) -> pd.DataFrame:
    """Per interviewer: interviews that start before the same interviewer's previous one that day had ended.

    One person cannot hold two interviews at once, so overlap means the times were not written as the interviews
    happened, or that more than one person worked under the name.
    """
    timed = df[(df["end"] > df["start"]) & (df["end"] - df["start"] <= 240)]
    timed = timed.sort_values(["interviewer", "village_id", "day", "start"])
    previous_end = timed.groupby(["interviewer", "village_id", "day"])["end"].transform(lambda s: s.cummax().shift())
    timed = timed.assign(overlap=timed["start"] < previous_end, minutes=timed["end"] - timed["start"])
    return timed.groupby("interviewer").agg(
        timed=("overlap", "size"),
        overlap_share=("overlap", "mean"),
        busiest_day=("day", lambda days: days.value_counts().max()),
        median_minutes=("minutes", "median"),
    )


def main(root: str) -> None:
    df = load(root)
    per_village = df.groupby("village_id")["interviewer"].nunique()
    per_interviewer = df.groupby("interviewer").size()
    print(
        f"{len(df):,} households, {df['village_id'].nunique()} villages, {len(per_interviewer)} interviewers "
        f"(median {per_interviewer.median():.0f} interviews; {per_village.median():.0f} interviewers per village)\n"
    )

    print("## percent_match within village (30 items; chance agreement from village marginals)")
    matches = percent_match(df, ITEMS, area="village_id", interviewer="interviewer", within_area=True)
    a = matches.attrs
    flagged = df.loc[matches["flagged"]]
    print(
        f"best match >= 0.85: {a['share_at_85']:.1%}; exact-duplicate pairs: {a['exact_duplicate_pairs']:,}; "
        f"beyond chance: {a['flagged']} (villages: {flagged['village_id'].value_counts().to_dict()})\n"
    )

    print("## exact twins by listing distance: other interviewer vs same interviewer")
    print(twins_by_distance(df).to_string(index=False), "\n")

    print(f"## interviewer_similarity (area = {SEGMENT}-household stretch of the listing)")
    similarity = interviewer_similarity(df, ITEMS, "interviewer", "segment", n_permutations=DRAWS, seed=1)
    a = similarity.attrs
    print(
        f"pooled excess {a['excess']:.4f} (p = {a['p_value']:.5f}); {a['single_interviewer_interviews']:.0%} of "
        f"interviews sit in single-interviewer stretches and are not tested; flagged {similarity['flagged'].sum()}"
    )
    columns = ["interviewer", "tested", "nearest_own", "nearest_other", "expected", "excess", "p_adjusted"]
    print(
        similarity.sort_values(["p_value", "excess"], ascending=[True, False]).head(8)[columns].to_string(index=False)
    )
    print()

    print("## pattern_repetition (nine hunger items; all-'never' strings excluded)")
    repetition = pattern_repetition(df, HUNGER, "interviewer", "segment", n_simulations=DRAWS, seed=1)
    columns = ["interviewer", "tested", "modal_pattern", "repeated_share", "expected_share", "p_adjusted"]
    print(f"flagged {repetition['flagged'].sum()}")
    print(repetition.sort_values("p_value").head(8)[columns].to_string(index=False), "\n")

    print("## straightlining (nine hunger items)")
    straight = straightlining(df, HUNGER, "interviewer", "segment")
    print(f"overall {straight.attrs['share']:.1%} straight-lined; flagged {straight['flagged'].sum()}")
    columns = ["interviewer", "tested", "share", "expected_share", "p_adjusted"]
    print(straight.sort_values("p_value").head(8)[columns].to_string(index=False), "\n")

    print("## interview clock (same interviewer, same village and day, overlapping start-end times)")
    times = clock(df)
    print(f"interviews overlapping an earlier one: {np.average(times['overlap_share'], weights=times['timed']):.1%}")
    print(times.sort_values("overlap_share", ascending=False).query("timed >= 100").head(8).to_string(), "\n")

    print("## by field team (wave-state): flags from the answers beside overlaps from the clock")
    flags = pd.DataFrame(
        {
            "tested": similarity.set_index("interviewer")["tested"] > 0,
            "similarity": similarity.set_index("interviewer")["flagged"],
            "repetition": repetition.set_index("interviewer")["flagged"],
            "straightlining": straight.set_index("interviewer")["flagged"],
        }
    ).join(times["overlap_share"])
    flags = flags[flags["tested"]]
    team = flags.index.str.rsplit("-", n=1).str[0]
    teams = flags.groupby(team).agg(
        interviewers=("tested", "size"),
        similarity=("similarity", "sum"),
        repetition=("repetition", "sum"),
        straightlining=("straightlining", "sum"),
        median_overlap=("overlap_share", "median"),
    )
    print(teams.sort_values("similarity", ascending=False).to_string())
    print(
        f"Spearman correlation, similarity excess vs clock overlap, across interviewers: "
        f"{similarity.set_index('interviewer')['excess'].corr(times['overlap_share'], method='spearman'):.2f}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
