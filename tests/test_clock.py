import numpy as np
import pandas as pd

from interviewer_checks import clock_flags, interview_clock, screen


def schedules(*, overlapper=None, seed=0, interviewers=20, days=10, per_day=6, slip=0.02):
    """Paper-style schedules: 40-minute interviews, a 20-minute walk between them, from 9:00.

    ``slip`` of interviews start 30 minutes early (an honest pen slip that creates an overlap). The ``overlapper``
    logs a new interview every 25 minutes, so each one starts before the last has ended.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for who in range(interviewers):
        gap = 25 if who == overlapper else 60
        for day in range(days):
            for slot in range(per_day):
                start = 9 * 60 + slot * gap - (30 if rng.random() < slip and slot > 0 else 0)
                rows.append({"interviewer": who, "day": day, "start": start, "end": start + 40, "area": who % 5})
    return pd.DataFrame(rows)


def test_overlapping_interviewer_is_flagged_and_nobody_else():
    df = schedules(overlapper=7)
    table = interview_clock(df, "interviewer", "start", end="end", date="day").set_index("interviewer")
    assert table.loc[7, "flagged"]
    assert table.loc[7, "overlap_share"] == 1.0
    assert table.drop(7)["flagged"].sum() == 0
    assert table.drop(7)["overlap_share"].max() < 0.15
    assert (table["max_per_day"] == 6).all()
    assert (table["median_minutes"] == 40).all()


def test_honest_back_to_back_schedules_raise_nothing():
    df = schedules(slip=0.0)
    df["end"] = df["start"] + 60
    table = interview_clock(df, "interviewer", "start", end="end", date="day", min_minutes=15)
    assert table["overlaps"].sum() == 0
    assert not table["flagged"].any()
    assert (table["short_share"] == 0).all()
    assert np.isnan(table.attrs["overlap_share"]) or table.attrs["overlap_share"] == 0


def test_interview_level_flags_with_datetimes():
    stamp = pd.to_datetime
    df = pd.DataFrame(
        {
            "who": ["a", "a", "a", "a", "a", "b", "a"],
            "start": stamp(
                ["2006-08-01 09:00", "2006-08-01 09:30", "2006-08-01 11:00", "2006-08-02 08:00", "2006-08-01 12:00"]
                + ["2006-08-01 09:10", "2006-08-01 09:52"]
            ),
            "end": stamp(
                ["2006-08-01 10:00", "2006-08-01 09:50", "2006-08-01 11:05", "2006-08-02 08:40", "2006-08-01 11:30"]
                + ["2006-08-01 09:50", "2006-08-01 10:20"]
            ),
        }
    )
    flags = clock_flags(df, "who", "start", end="end", min_minutes=10)
    # The last interview starts after the 9:30 one ended but before the 9:00 one did.
    assert flags["overlap"].tolist() == [False, True, False, False, False, False, True]
    assert flags["short"].tolist() == [False, False, True, False, False, False, False]
    assert flags["timed"].tolist() == [True, True, True, True, False, True, True]
    assert flags["follows"].tolist() == [False, True, True, False, False, False, True]
    assert flags["flagged"].tolist() == [False, True, True, False, False, False, True]

    df["length"] = flags["minutes"]
    by_duration = clock_flags(df, "who", "start", duration="length", min_minutes=10)
    pd.testing.assert_frame_equal(by_duration, flags)

    table = interview_clock(df, "who", "start", end="end", min_minutes=10).set_index("who")
    assert table.loc["a", "overlaps"] == 2 and table.loc["a", "follows"] == 3
    assert table.loc["a", "max_per_day"] == 5
    assert table.loc["a", "short_share"] == 0.2


def test_screen_adds_the_clock_only_when_times_are_given():
    df = schedules(overlapper=7)
    df["treated"] = np.arange(len(df)) % 2
    without = screen(df, "interviewer", treatment="treated")
    assert "interview_clock" not in without.results and "flag_clock" not in without.table
    result = screen(df, "interviewer", treatment="treated", start="start", end="end", date="day", min_minutes=15)
    assert "interview_clock" in result.results
    table = result.table.set_index("interviewer")
    assert table.loc[7, "flags"] == "clock"
    assert table.drop(7)["flag_clock"].sum() == 0
    assert "Clock:" in result.summary


def test_flags_follow_rows_not_index_labels():
    df = schedules(overlapper=7, interviewers=10)
    shuffled = df.sample(frac=1, random_state=0)
    expected = clock_flags(shuffled, "interviewer", "start", end="end", date="day")
    relabelled = clock_flags(
        shuffled.set_axis(np.zeros(len(df), dtype=int)), "interviewer", "start", end="end", date="day"
    )
    assert (relabelled["overlap"].to_numpy() == expected["overlap"].to_numpy()).all()
    assert expected.loc[shuffled["interviewer"] == 7, "overlap"].mean() > 0.8
