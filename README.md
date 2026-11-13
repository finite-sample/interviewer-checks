# interviewer-checks

[![PyPI version](https://img.shields.io/pypi/v/interviewer-checks.svg)](https://pypi.org/project/interviewer-checks/)
[![CI](https://github.com/finite-sample/interviewer-checks/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/interviewer-checks/actions/workflows/ci.yml)

Find data problems tied to interviewers in face-to-face surveys and survey experiments.

An interviewer who invents interviews, copies them, or skips the randomization leaves traces. Three are common:
estimates that depend on a handful of interviewers, treatment effects that differ by interviewer more than sampling
noise allows, and answers that repeat within an interviewer's work more than local opinion explains. When the
schedules record start and end times, there is a fourth: interviews that overlap the same interviewer's previous one,
which one person cannot do. The package looks for those generic traces rather than for any particular pattern. You don't need to know what the problem looks
like before you find it.

Data go in and come out as pandas data frames. Every check returns one row per interviewer (or per interview), and
its scalar results sit in `.attrs`.

## Install

From a clone of the repository:

```bash
uv pip install -e .
```

## The checks

| Function | What it compares | Null | Catches |
|---|---|---|---|
| `cluster_influence(df, estimate_fn, cluster)` | the estimate with each cluster left out against the full-data estimate | none (descriptive; shifts are scaled by the jackknife SE) | estimates that rest on a few interviewers |
| `effect_heterogeneity(df, outcome, treatment, cluster, respondent=None, method="auto")` | each interviewer's treated-minus-control difference against the pooled one, with respondent-clustered SEs in long format | Cochran's Q across interviewers and per-interviewer z, judged by shuffling treatment within interviewer (max-T for the flags) when any arm has under 100 units, else chi-square and normal with Bonferroni | interviewer-specific treatment effects; interviewers with zero standard error |
| `assignment_balance(df, treatment, cluster)` | each interviewer's treated count against the overall share | exact binomial, with Bonferroni; chi-square of interviewer × arm | interviews that skipped the randomization |
| `percent_match(df, items, area=None)` | each interview's best match (share of identical answers) against the agreement expected by chance from item distributions in its area | items agree independently with those chances; Bonferroni over all pairs searched | copied or duplicated interviews |
| `interviewer_similarity(df, items, interviewer, area)` | how close each interview's nearest twin among the same interviewer's work is, against shuffled interviewer labels | shuffle interviewer labels within area | interviewers whose interviews resemble each other |
| `pattern_repetition(df, items, interviewer, area, strata=None)` | how many of an interviewer's interviews carry their most common full answer string, by stratum (e.g. arm) | shuffle interviewer labels within area × stratum | interviewers who write the same answers over and over |
| `straightlining(df, items, interviewer, area)` | each interviewer's count of interviews that give one answer to every item | exact: a sum over areas of hypergeometric counts (the shuffle null in closed form) | interviewers who take the shortcut of one answer to everything |
| `interview_clock(df, interviewer, start, end=None, duration=None, date=None)` | each interviewer's share of interviews that start before their own previous interview that day ended, against the survey-wide rate; also the busiest day, median length and share below a floor | exact one-sided binomial, with Bonferroni; the null is that every interviewer makes clock errors at the same rate | times that cannot all be true; two people under one id |
| `clock_flags(df, interviewer, start, ...)` | the same comparison, one row per interview (`overlap`, `short`, `flagged`) | none (each overlap is a logical impossibility) | the rows behind `interview_clock` |
| `screen(df, interviewer, ...)` | runs whichever checks the columns allow (the clock only when `start` is passed) | | one table with `flag_*` columns, plus a text summary |

Each docstring spells out the mechanism, so `help(interviewer_checks.percent_match)` says exactly what is compared.

### Three design choices that matter

**Nulls condition on area.** People who live near each other answer alike. If interviewers work in different areas,
"interviewer X's answers are unusual" may just mean that X works somewhere unusual. So every interviewer-level null
shuffles interviewer labels *within* an area and keeps how many interviews each interviewer did there. That way an
interviewer is compared only with neighbors who interviewed the same population.

The weak point is that an area worked by a single interviewer can't be shuffled. There, interviewer and area are the
same thing, and no data-only check can tell them apart. Each check reports `single_interviewer_areas` and
`single_interviewer_interviews` in `.attrs` and leaves those interviews out. If most of your sampling points have one
interviewer (common: one interviewer per cluster is a standard field design), pass a coarser area such as the district.
Then remember that the comparison also absorbs differences between villages inside the district. The case study below
shows what that costs.

**Interviewer effects are judged by permutation when arms are small.** Cochran's Q adds up each interviewer's
squared z, and each z divides by a standard error estimated from that interviewer's own few respondents. A noisy
standard error is sometimes far too small, so the squared z has a heavier tail than the chi-square assumes, and the
excess adds up over interviewers. With a common effect, 4 respondents per arm and 30 interviewers, Q against
chi-square rejects at 5% in 73% of simulated surveys; with 10 per arm and 60 interviewers, in 34%; even with 50 per
arm it rejects in 7-9%. `effect_heterogeneity` therefore removes the estimated common effect and shuffles treatment
labels within each interviewer, keeping each interviewer's number of treated and control respondents, and recomputes
Q and every z on each shuffle. In the same simulations that rejects in 4-6%. The flags compare each interviewer's
|z| with the largest |z| in each shuffle (Westfall and Young's max-T), which holds the chance of any false flag near
5% (6% against Bonferroni's 62% at 4 per arm). `method="auto"` uses this whenever some tested interviewer has fewer
than 100 respondents in an arm, which in practice is most surveys; `attrs["method"]` says which ran. The shuffle tests
that treatment shifts every respondent's outcome by the same amount. If treatment also changes how spread out the
answers are, it over-rejects with very few respondents per arm (15-30% in simulation when the treated SD is twice the
control SD with 2-8 per arm), though far less than chi-square does (80-95%).

**Percent match is judged against expected agreement, not against 85%.** Kuriakose and Robbins flag respondents who
match another respondent on 85% or more of the questions. Simmons, Mercer, Schwarzer and Kennedy (2016) showed that
raw percent match depends on the number of items, the number of categories, how lopsided each item is, and how many
partners there are to match against. Take eight yes/no items that 90% of people answer "yes": two strangers then agree
on an item 0.9² + 0.1² = 82% of the time, and nearly every respondent has a best match of at least 0.85. The test
suite checks this case: `share_at_85 > 0.95`, `expected_match ≈ 0.82`, zero flags. `percent_match` therefore assigns
each item a chance-agreement probability from its answer distribution in the pair's area. Within an area this is
Σ c(c − 1) / (n(n − 1)), the share of pairs of distinct interviews that agree, which removes the small-sample upward
bias of Σ p̂². The function then gives the exact probability of the observed number of agreements, multiplied by the
number of pairs searched. One assumption remains: items are treated as independent. Real attitudes are correlated, so
genuine respondents agree more often than that predicts. In simulation, with independent items nothing is flagged;
with items that share a respondent trait, about 1% of honest interviews are flagged. `screen` therefore reports
`percent_match` counts per interviewer but does not raise a flag on them. The interviewer-level question, "do close
matches concentrate within interviewers?", goes to `interviewer_similarity`, whose shuffle null makes no independence
assumption.

## Worked example (synthetic)

There are 60 interviewers in 20 areas, with 120 interviews each, and a four-item outcome battery. Interviewer 12
invents 60% of their interviews. Most of the invented interviews are recorded as "control" and answered 4-0-0-4; the
rest are recorded as "treated" and answered 4-1-0-4. The full script is [`examples/synthetic.py`](examples/synthetic.py).

```python
result = screen(
    df,
    "interviewer",
    pattern_items=["p1", "p2", "p3", "p4"],
    area="area",
    treatment="treated",
    outcome="support",
)
print(result.summary)
print(result.table.head(4)[["interviewer", "effect_z", "share_treated", "repeated_share", "flags"]])
```

```
60 interviewers, 7,200 interviews.
Areas (area): 20, 0% worked by a single interviewer.
Pooled effect -0.028 (SE 0.004; 0.005 clustered by interviewer). Heterogeneity across interviewers: Q = 99.6 on 59 df, p = 0.0021 (permutation). Interviewer effects off the pooled one: 12. No variation in the outcome: none.
Leaving one interviewer out moves the effect most for 12 (-0.0043, -0.8 jackknife SE), 22 (-0.0013, -0.2 jackknife SE), 14 (+0.0010, +0.2 jackknife SE).
Assignment: 49.5% treated overall; interviewer x treatment chi2 = 68.4 on 59 df, p = 0.188. Lopsided interviewers: none.
Repeated answer strings beyond what the area produces: 12.
Straight-lining (9.3% of interviews overall) above the area rate: none.
Flagged by two or more checks: 12.
  interviewer  effect_z  share_treated  repeated_share               flags
0          12  6.140406       0.350000        0.583333  effect, repetition
1           0  1.873818       0.508333        0.075000                    
2           1 -1.917643       0.508333        0.075000                    
3           2 -1.006649       0.558333        0.091667                    
```

The planted interviewer is the only one flagged twice, and the only one flagged at all by the effect check. (Under
the chi-square reference with Bonferroni, interviewer 22 was flagged as well, a false positive; any reference holds the
chance of a false flag to about 5% per check, not zero.) Interviewer 12's 35% treated share over 120
interviews is not enough for the balance check after correcting for 60 interviewers. A single flag is a lead to
follow up, not a finding.

## Case study: Blair, Fair, Malhotra and Shapiro (2013)

Blair et al. ran a 6,000-person endorsement experiment in Pakistan. Each respondent rated four policies, either
unendorsed (control) or endorsed by a militant group. Treatment was randomized within interviewer, and the
replication file records the interviewer. [`examples/blair_pakistan.py`](examples/blair_pakistan.py) reads it. The
items are `q800`, `q810`, `q830` and `q840`, with suffixes `a` (control) through `e`. They are coded 1 = a great deal
through 5 = not at all, with 98 and 99 for don't know and no response, and the script recodes them to 0-4 with 4 as
the most support. The interviewer is `name`, the 12-respondent sampling point is `spn`, and the district is `d24`.
The data are not distributed with this package.

```bash
uv run python examples/blair_pakistan.py path/to/fms09-data-weighted-20june.dta
```

Output (verbatim):

```
5,444 respondents, 75 interviewers

## effect_heterogeneity (long format, SEs clustered on respondent)
pooled ATE -0.013 (SE 0.005); Cochran's Q = 118.0 on 59 df, p = 0.0003 (permutation, smallest arm 7); zero standard error: [33, 35]
 interviewer  respondents  treated  control  effect    se     z   p_value  p_adjusted  flagged
          35           12        4        8   0.000 0.000   inf       NaN         NaN    False
          33           12        4        8   0.000 0.000   inf       NaN         NaN    False
          71           51       17       34   0.057 0.010 7.096 5.000e-05   1.000e-04     True
          37          160       60      100   0.082 0.024 4.002 4.000e-04   2.995e-02     True
           7          101       48       53   0.037 0.021 2.349 2.290e-02   8.134e-01    False
without 7, 37, 71: ATE -0.021; Q = 52.6 on 56 df, p = 0.754 (permutation)

## pattern_repetition (strings over the four policies, by arm, shuffled within district)
 interviewer  tested  repeated_share  expected_share  p_adjusted                     modal_by_stratum
          71      51           0.843           0.324       0.008 0: 4-0-0-4 (0.94); 1: 4-1-0-4 (0.65)
           7     101           0.594           0.193       0.008 0: 4-0-0-4 (0.58); 1: 4-1-0-4 (0.60)
          24      33           0.576           0.207       0.008 0: 4-3-3-3 (0.71); 1: 4-3-3-3 (0.44)
          35      12           1.000           0.648       0.345 0: 3-4-3-4 (1.00); 1: 3-4-3-4 (1.00)
          37     160           0.569           0.237       0.008 0: 4-0-0-4 (0.58); 1: 4-1-0-4 (0.55)
          29      22           0.409           0.138       0.008 0: 4-2-2-3 (0.40); 1: 4-2-2-3 (0.42)
          72       6           0.500           0.282       1.000 0: 4-3-3-0 (0.67); 1: 4-4-3-0 (0.33)
          67      12           0.417           0.240       1.000 0: 3-2-3-2 (0.38); 1: 3-3-2-1 (0.50)
flagged: [7, 10, 12, 22, 24, 29, 37, 41, 44, 45, 55, 56, 61, 63, 66, 71]

## screen (one row per interview; outcome = mean support over the four policies)
75 interviewers, 5,444 interviews.
Areas (district): 50, 0% worked by a single interviewer.
Pooled effect -0.013 (SE 0.005; 0.007 clustered by interviewer). Heterogeneity across interviewers: Q = 118.0 on 59 df, p = 0.00025 (permutation). Interviewer effects off the pooled one: 37, 71. No variation in the outcome: 33, 35.
Leaving one interviewer out moves the effect most for 37 (-0.0047, -0.7 jackknife SE), 66 (+0.0022, +0.3 jackknife SE), 71 (-0.0018, -0.3 jackknife SE).
Assignment: 49.2% treated overall; interviewer x treatment chi2 = 76.3 on 74 df, p = 0.404. Lopsided interviewers: none.
Repeated answer strings beyond what the area produces: 7, 10, 12, 22, 24, 29, 37, 41, 44, 45, 46, 55, 56, 61, 63, 66, 71.
Straight-lining (21.1% of interviews overall) above the area rate: 3, 11, 15, 23, 26, 32, 33, 41, 57, 60, 62, 66.
Flagged by two or more checks: 33, 37, 41, 66, 71. 

interviewer  interviews  effect_z  ate_shift  share_treated  repeated_share                         flags
         33          12       inf  3.560e-04          0.333           0.000 zero_variance, straightlining
         37         160     4.002 -4.667e-03          0.375           0.569            effect, repetition
         41         192     0.655 -6.466e-04          0.516           0.260    repetition, straightlining
         66         174    -2.139  2.161e-03          0.483           0.092    repetition, straightlining
         71          51     7.096 -1.823e-03          0.333           0.843            effect, repetition
          3         279    -0.497 -1.338e-03          0.523           0.014                straightlining
          7         101     2.349 -1.142e-03          0.475           0.594                    repetition
         10          60     0.744 -5.546e-05          0.400           0.183                    repetition 

## percent_match over the whole questionnaire (chance agreement from sampling-point marginals)
92 items; best match >= 0.85: 40.5%; exact-duplicate pairs: 175; beyond chance (independent items): 940
same search within sampling points: best match >= 0.85: 31.4%; exact-duplicate pairs: 79
```

What the generic checks find, and what they miss:

- **Treatment-effect heterogeneity is the strongest signal.** Interviewers differ in their effects more than noise
  allows (Q = 118.0 on 59 df). The smallest tested interviewer has 7 respondents in an arm, so the test shuffles
  treatment within interviewer; 4 of 19,999 shuffles reach the observed Q (p = 0.0003). The two most extreme
  interviewers, 71 and 37, survive the max-T correction over 60 interviewers (adjusted p = 0.0001 and 0.03). Interviewer 7 is next (z = 2.3, unadjusted p = 0.02), but not once you correct for looking at 60
  interviewers. Without 7, 37 and 71, the pooled effect moves from −0.013 to −0.021 and the heterogeneity disappears
  (Q = 52.6 on 56 df, p = 0.76). Interviewers 33 and 35 have zero standard error: every respondent's answers equal
  their arm's mean, so neither interviewer has any respondent-to-respondent variation. Both runs drop the 35
  respondents who got both endorsed and unendorsed items, because they belong to neither arm and the shuffle moves
  whole respondents. Keeping them lowers Q to 93.1, almost entirely because of one respondent of interviewer 71,
  recorded as 4-1-0-4 with one policy unendorsed: interviewer 71's other answers vary so little that this one
  interview raises their standard error by 37%. The paper's R code gives the same Q and a matching permutation
  reference (8 of 20,000 reshuffles reach 118.0 in both implementations).
- **Repeated strings point at the same three interviewers but are not specific here.** 71, 7 and 37 record 4-0-0-4 in
  58–94% of their control interviews and 4-1-0-4 in 55–65% of their treated ones. No other interviewer has
  arm-specific strings like these. Still, 16 interviewers are flagged, because the shuffle is within district while
  485 of the 500 sampling points have a single interviewer. "Same district" is not "same village", so an interviewer
  whose village answers alike looks repetitive. Interviewers 24 and 29 are examples: they repeat one string in both
  arms. Treat this check as a ranking here. Its distinctive output is `modal_by_stratum`: a string that changes with
  the arm is what moves an effect.
- **Influence points at 37.** Dropping 37 moves the pooled effect more than dropping anyone else (−0.0047). The
  shift is under one jackknife SE, because no single interviewer dominates. It is the three together that pull the
  effect from −0.021 to −0.013.
- **Balance does not flag anyone individually.** Together, the three interviewers recorded 188 control and 125
  treated interviews (exact binomial p = 0.001 against the 49.2% overall share). One at a time, none of them is lopsided enough to survive correction for
  75 interviewers. The check is weak against moderate skew. This is the one place where pooling the suspects, a
  confirmatory step after the generic checks have named them, adds evidence.
- **Whole-questionnaire matches are high but don't isolate interviewers.** 31% of respondents have a best match at or
  above 0.85 within their sampling point, and 79 pairs are exact duplicates. This matches the manuscript's numbers.
  Because sampling points nest within interviewers, `interviewer_similarity` cannot separate "copied by the
  interviewer" from "same village", and with the district as area it inherits the same caveat as pattern repetition.

## Case study: the REDS 2013-14 household listing

NCAER's Rural Economic and Demographic Survey (REDS), in its 2013-14 SEPRI rounds, listed 92,996 households in 193
villages across 13 states on paper schedules. [`examples/reds.py`](examples/reds.py) reads sections 1-3 from wherever
the data sit; they are not distributed with this package.

```bash
uv run python examples/reds.py path/to/reds/data   # about 12 minutes on a laptop
```

Three choices shape the results, and each is open to dispute:

- **Who the interviewer is.** The schedules record `interviewer_name` as typed text, so one person appears under
  many spellings (KASHI NATH, KASHINATH, KAHSI NATH): 5,605 raw names for about 600 people. Within each village the
  script maps every name to the most similar frequent name there (difflib ratio of at least 0.8), and drops the 1.8%
  of households whose name matches none. The result is 598 interviewers with a median of 120 interviews and about 6
  interviewers per village.
- **The items.** There are 30 closed-ended items: dwelling (roof, floor, water, toilet, electricity), food security
  (including nine never/rarely/sometimes/often hunger items), and the head's caste group, religion and job card.
- **The area.** Interviewers share villages, which is the design these checks need, but within a village each one
  took short runs of the street-ordered listing, about six households at a time. Own interviews are therefore nearer
  neighbors than other interviewers' interviews in the same village. The area is a 50-household stretch of the
  listing, so the shuffle compares an interviewer only with colleagues on the same few streets. That narrows the
  geographic gap but cannot close it.

Output (verbatim, abridged to the top rows of each table):

```
91,338 households, 191 villages, 598 interviewers (median 120 interviews; 6 interviewers per village)

## percent_match within village (30 items; chance agreement from village marginals)
best match >= 0.85: 97.3%; exact-duplicate pairs: 279,583; beyond chance: 23 (villages: {'Sepri2-12-74-175': 23})

## exact twins by listing distance: other interviewer vs same interviewer
distance  pairs_other  twins_other  pairs_same  twins_same  ratio
       1         7657        0.027       83490       0.074  2.777
       2        11699        0.021       79257       0.053  2.584
     3-5        56213        0.014      215509       0.039  2.853
    6-10       136178        0.011      312872       0.028  2.458
   11-25       537112        0.010      781508       0.019  1.954
   26-50      1192909        0.008      909846       0.013  1.698
  51-100      3004279        0.007      847166       0.010  1.545
    >100     34976696        0.005     2113724       0.004  0.945

## interviewer_similarity (area = 50-household stretch of the listing)
pooled excess 0.0157 (p = 0.00002); 29% of interviews sit in single-interviewer stretches and are not tested; flagged 141

## pattern_repetition (nine hunger items; all-'never' strings excluded)
flagged 65

## straightlining (nine hunger items)
overall 65.2% straight-lined; flagged 20
                 interviewer  tested  share  expected_share  p_adjusted
      Sepri2-15-SANATANPANDA     302  0.602           0.331   4.696e-36
     Sepri2-12-DASORATHGHOSH     242  0.908           0.719   6.071e-32
       Sepri1-17-GKMAHADEVAN     130  0.557           0.290   1.112e-25

## interview_clock (same interviewer, same village and day; over 4 hours counts as untimed)
interviews that start before the same interviewer's previous one that day ended: 8.2% of those with a previous one; under 10 minutes: 0.1%; flagged 66
                 interviewer  timed  overlap_share  max_per_day  median_minutes  p_adjusted
Sepri2-15-SURYAMANIMAHAPATRA    231          0.687         26.0            80.0   3.809e-94
     Sepri2-15-DAMODARBEHERA    435          0.679         28.0            60.0  1.754e-181
          Sepri1-17-SARUNRAJ    303          0.568         35.0            60.0   6.567e-86

## by field team (wave-state): flags from the answers beside overlaps from the clock
             interviewers  similarity  repetition  straightlining  median_overlap
interviewer                                                                      
Sepri1-17              55          28          32               4           0.057
Sepri1-5               35          24          11               3           0.060
Sepri1-14              33          18           6               1           0.100
Sepri2-12              49          18           9               4           0.103
Sepri1-13              26          16           0               2           0.153
Sepri1-11              23          15           1               2           0.110
Sepri2-15              36           9           6               1           0.125
Sepri1-6               84           6           0               0           0.009
Sepri1-7               48           3           0               2           0.014
Sepri2-10              80           3           0               0           0.014
Sepri1-16              58           1           0               1           0.016
Sepri2-3               40           0           0               0           0.006
Sepri2-4               28           0           0               0           0.011
Spearman correlation, similarity excess vs clock overlap, across interviewers: 0.29
```

What this shows:

- **Raw percent match is uninformative here.** 97% of households have a best match of at least 0.85, and 279,583
  pairs agree on all 30 items, because neighboring households in a village share housing, water, caste and (mostly)
  no hunger. Judged against expected agreement, only 23 households stand out, all in one village. Their rows look
  genuine rather than copied: the first pair has consecutive listing numbers and was interviewed back to back by
  the same interviewer (9:00-9:30, 9:35-10:00). That is what two households in one compound look like.
- **Twins concentrate within interviewers at a fixed distance.** Households a few listing places apart are 2.5-2.9
  times as likely to be exact twins when one interviewer did both. The excess shrinks with distance and vanishes
  beyond 100 places, so it is not a fixed interviewer style. It is either geography that the listing distance
  misses (runs ending at lane boundaries) or interviewers filling in neighboring households from one visit.
  Among pairs within ten places, two interviewers working the same day produce twins 0.9% of the time; one
  interviewer on one day produces them 5.1% of the time.
- **The answer checks and the clock point to the same field teams.** The clock check shares no code or data with
  the answer checks. It counts interviews that begin before the same interviewer's previous interview that day has
  ended. One person cannot do two interviews at once, so either the times were not written down as the interviews
  happened or more than one person worked under the name. Seven state teams (wave-state codes 17, 5, 14, 12, 13,
  11, 15) hold 128 of the 141 similarity flags and 65 of the 65 repetition flags. Their median interviewer overlaps
  the previous interview in 6-15% of interviews that follow another the same day. The other six teams have 13
  similarity flags and a median overlap of 0.6-1.6%. `interview_clock` flags 66 interviewers whose overlap rate is
  above the survey-wide 8.2%. One row-level example: on its busiest
  day SARUNRAJ logs 35 interviews. Five consecutive households share the slot 07:10-07:50, the others start on
  whole hours with two or three per hour from 07:00 to 20:00, and the food-security answers barely vary.
- **Straight-lining is mostly genuine but uneven.** 65% of households answer "never" to all nine hunger items. Twenty
  interviewers record that far more often than colleagues on the same streets (SANATANPANDA: 60% against 33%).
  Food-insecurity estimates from these data inherit that difference.

What this does not show: that any interview was invented. The geographic confound is narrowed, not removed, and an
interviewer name may stand for a pair working together. What it does show is where to look. A few field teams
produce answers that are more alike within interviewers, and times that cannot all be true. An analysis that uses
these data can rerun its estimates without those teams, or with interviewer fixed effects.

No published estimate on this machine can be checked against these flags. Munshi and Rosenzweig (2016, *AER*,
openICPSR 112956) use the 2006 REDS listing. That round was collected by different field staff, so flags in the
2013-14 rounds say nothing about it. Jensenius (2015, *AEJ: Applied*, openICPSR 113613) uses census village data.

## Case study: Beaman, Duflo, Pande and Topalova (2012)

West Bengal reserves the post of village-council head (pradhan) for a woman in a rotating, effectively random set
of councils. Beaman et al. surveyed 3,680 adolescents in 165 councils of Birbhum district in 2006 and report that,
in councils reserved twice, the gap between girls' and boys' aspirations shrank by 0.166 standard deviations
(Table 2, SE 0.057). [`examples/beaman_west_bengal.py`](examples/beaman_west_bengal.py) reads J-PAL's public files
(Harvard Dataverse, doi:10.7910/DVN/O3UKFO); they are not distributed with this package.

```bash
uv run python examples/beaman_west_bengal.py path/to/beaman/data   # about 10 seconds
```

What the data hold, and the choices that follow:

- **Interviewers.** The adolescent schedule records an interviewer code (`AA0_9_1`): 27 codes, 19 with 80-280
  interviews and 8 with fewer than 20. The household survey uses a separate set of codes (1-55), so the two
  surveys' interviewers cannot be linked. The design nests block (19) > council (`AA0_2b`, 162 matched) > village
  (council plus `AA0_3b`, 490). Interviewers cross blocks and councils (a median council has two), but 87% of
  villages were worked by one interviewer.
- **Outcomes.** The paper's normalized index averages four yes/no aspirations, each standardized by the
  never-reserved mean and SD. Three follow the earlier replication in `quota_aspirations`: not wanting to be a
  housewife or do what in-laws want (`B1_3`), wanting to marry after 18 (`B1_2`), wanting to graduate (`B1_1`). The
  fourth, wanting a job that needs an education, has no code. The script reads it from the free-text occupation:
  teacher, doctor, engineer, professor, lawyer, IPS, officer, manager. That list reproduces the paper's
  never-reserved means (0.154 and 0.162 against 0.157 and 0.166). Adding nurses raises the girls' share to 0.21, so
  the paper probably left them out. 38 adolescents have no recorded sex and are dropped.
- **Estimate.** The paper's coefficient comes from an OLS regression with village controls. The script takes the
  saturated difference in differences of cell means, which is that regression without the controls. Its standard
  error is a delete-one-council jackknife (`cluster_influence` over councils).
- **Parents.** The paper's Table 1 (parents' aspirations, 0.136) uses a module that is not in the public files.

Output (verbatim, with the clock table abridged):

```
3,642 adolescents, 19 blocks, 162 councils, 490 villages, 27 interviewers (median 168 interviews; 8 with fewer than 20)
councils by arm: {'never': 72, 'once': 70, 'twice': 20}; villages with one interviewer: 87%; councils with one interviewer: 23%

## replication: never-reserved means and the twice-reserved change in the girl-boy gap (Table 2)
       outcome  boys_paper  boys  girls_paper  girls  change_paper  change  se_paper  se_gp_jackknife
  no_housewife       0.998 0.998        0.600  0.580         0.075   0.094     0.042            0.044
  educated_job       0.157 0.154        0.166  0.162         0.099   0.102     0.051            0.048
marry_after_18       0.980 0.980        0.660  0.656         0.085   0.092     0.049            0.052
      graduate       0.296 0.290        0.195  0.190         0.001   0.017     0.045            0.047
         index       0.248 0.252       -0.260 -0.261         0.166   0.188     0.057            0.057

## assignment_balance (one row per interviewer x council; twice-reserved against never-reserved)
23.5% of interviewer x council pairs twice-reserved; chi2 = 30.9 on 24 df, p = 0.155; flagged 0
interviewer  units  treated  share_treated  expected_treated  p_value  p_adjusted
         61      5        4          0.800             1.176    0.012       0.311
         96     11        0          0.000             2.588    0.078       1.000
        100      8        0          0.000             1.882    0.211       1.000
         79      9        4          0.444             2.118    0.229       1.000
        109      1        1          1.000             0.235    0.235       1.000
ever reserved against never, all councils: 57.6%; chi2 = 28.4 on 26 df, p = 0.340

## effect_heterogeneity (girl-boy gap in the index per interviewer x council; twice vs never)
194 cells; 12 interviewers with at least two cells in each arm (smallest arm 2). Pooled change 0.258 (SE 0.112; 0.110 clustered by interviewer); Cochran's Q = 32.1 on 11 df
against chi-square: p = 0.001, flagged 1; against 9,999 relabelings within interviewer (permutation, chosen by method='auto'): p = 0.196, flagged 0
 interviewer  treated  control  effect    se      z  p_value_chi2  p_adjusted_chi2  p_value_perm  p_adjusted_perm
          81        3        6   1.056 0.211  3.785     1.537e-04            0.002         0.055            0.258
          83        4        7  -0.517 0.284 -2.724     6.441e-03            0.077         0.017            0.545
          80        3        5   0.044 0.121 -1.762     7.807e-02            0.937         0.225            0.901
         107        3        4  -0.217 0.274 -1.731     8.345e-02            1.000         0.184            0.911
         103        3        5   0.618 0.310  1.163     2.450e-01            1.000         0.357            0.992

## cluster_influence (headline change in the gap, each interviewer left out)
full 0.188; interviewer-jackknife SE 0.073; council-jackknife SE 0.057; range without one interviewer 0.159 to 0.246
interviewer  rows  estimate_without  shift  shift_in_se
         83   191             0.246  0.058        0.799
         81   197             0.159 -0.029       -0.397
        107    97             0.206  0.018        0.253
         98   283             0.169 -0.018       -0.250
         68   247             0.173 -0.014       -0.197

## percent_match within village (31 closed-ended items; chance from village marginals)
best match >= 0.85: 9.4%; exact-duplicate pairs: 1; beyond chance (independent items): 16
flagged interviews by interviewer: {96: 4, 103: 2, 79: 2, 110: 2, 106: 2, 101: 2, 80: 2}

## interviewer_similarity (area = village)
pooled excess +0.0255 (p = 0.0001); 84% of interviews in single-interviewer villages are not tested; flagged 4
interviewer  tested  nearest_own  nearest_other  excess  p_adjusted
         84      31        0.798          0.601   0.048       0.002
         79      30        0.735          0.593   0.047       0.012
        102      46        0.696          0.633   0.036       0.024
         68      45        0.719          0.597   0.042       0.038
         81      19        0.729          0.635   0.070       0.185

## interviewer_similarity (area = gp)
pooled excess +0.0333 (p = 0.0001); 21% of interviews in single-interviewer gps are not tested; flagged 14
interviewer  tested  nearest_own  nearest_other  excess  p_adjusted
        109      13        0.787          0.680   0.087       0.003
        106      96        0.803          0.708   0.068       0.003
         84      83        0.742          0.667   0.065       0.003
         82     122        0.778          0.680   0.057       0.003
         81     151        0.771          0.693   0.055       0.003

## interview_clock (same interviewer and day; 5 minutes or less is short; over 4 hours untimed)
timed 99.3%; overlapping the previous interview that day: 3.5%; 5 minutes or less: 12.4%; flagged 0; start and end minutes on a multiple of 5: 82%, 80%
 interviewer  interviews  timed  follows  overlaps  overlap_share  max_per_day  median_minutes  short_share  p_value  p_adjusted  flagged
          84          83     82       47         6          0.128          8.0            15.0        0.000    0.006       0.147    False
         101         206    204      128        11          0.086         10.0            10.0        0.049    0.005       0.139    False
          82         168    166      100         6          0.060          7.0            10.0        0.072    0.137       1.000    False
          80         228    228      137         8          0.058          7.0            15.0        0.044    0.107       1.000    False
         104         211    209      128         7          0.055          7.0             5.0        0.603    0.160       1.000    False
         ...
         110         138    137       81         2          0.025          6.0             5.0        0.788    0.778       1.000    False
         ...
interviewers with most interviews at 5 minutes or less: [104, 110]; headline change without them: 0.190 (3,293 rows)
```

What this shows:

- **The replication is close but not exact.** The never-reserved means match the paper to within 0.02, and the
  standard error matches (0.057). The headline change is 0.188 against the published 0.166, a gap of 0.4 SE. The
  likely sources are the missing village controls and the free-text housewife coding (girls 0.580 against 0.600).
  So the variables are right; the point estimate is a little larger without controls.
- **(a) Interviewers are balanced across arms.** Treatment is assigned to councils, so each interviewer x council
  pair is one unit. No interviewer's share of twice-reserved councils survives correction (lowest adjusted p 0.31),
  and the interviewer x arm table is unremarkable (p = 0.16; p = 0.34 for ever- against never-reserved). Two large
  interviewers (96 with 11 councils, 100 with 8) never worked a twice-reserved council. That fits chance, given
  that only 20 councils were reserved twice. It does mean those interviewers contribute nothing to the within-
  interviewer contrast.
- **(b) No reliable interviewer heterogeneity, and no interviewer carries the result.** Against chi-square, Q = 32.1
  on 11 df gives p = 0.001 and interviewer 81 survives Bonferroni (z = 3.8). Both rest on standard errors computed
  from two to four councils per arm, which are too noisy for those references; the simulations above show how far
  they over-reject in this regime. With so few councils, `method="auto"` shuffles which of an interviewer's councils
  count as twice-reserved, with the common effect removed. Then Q has p = 0.20 (the shuffled Q averages 23.5 against
  a chi-square mean of 11), and interviewer 81 has p = 0.055 alone and 0.26 after max-T, so nothing is flagged. The
  shuffle assumes treatment shifts outcomes without changing their spread; here the within-interviewer spread of the
  gap is the same in both arms (SD 0.487 twice-reserved, 0.485 never). Leaving out any one interviewer
  keeps the headline between 0.159 and 0.246, at least 2.8 council-jackknife SEs from zero. The largest shifts are 83
  (+0.058) and 81 (−0.029), and they pull in opposite directions.
- **(c) Answer similarity is weak, and where it is testable it is not specific.** Within villages, 9.4% of
  adolescents have a best match of 0.85 or more over 31 items, only one pair is an exact duplicate, and 16 interviews
  match beyond chance, spread over seven interviewers. `interviewer_similarity` is the binding constraint: 87% of
  villages had one interviewer, so a within-village comparison tests only 16% of interviews. It flags 4 interviewers
  (84, 79, 102, 68). With the council as the area it tests 79% of interviews and flags 14 of 27. That comparison also
  absorbs village differences inside a council, so it is a ranking, as with the district-level run on the Pakistan
  data, not evidence of copying.
- **(d) The clock is mostly consistent, but it is coarse.** Of interviews that follow another the same day, 3.5%
  start before the previous one ended; no interviewer exceeds that rate after correction (84 is highest at 12.8%,
  adjusted p 0.15). Times are mostly rounded to five minutes (82% of start minutes), so the check cannot see
  overlaps shorter than that. The distinctive pattern is length. Interviewers 104 and 110 log 60% and 79% of their
  interviews at five minutes or less (median 5), against a survey median of 10. The schedule has at least 31 closed
  questions plus the occupation and time-use follow-ups, so that is fast but not impossible, and the heaping suggests lengths were written from memory. Dropping both
  leaves the headline at 0.190.

What this does not show: that anything in these data was invented. On this evidence, the twice-reserved result does
not rest on any one interviewer, interviewers were not steered toward or away from reserved councils, and the
clocks raise only two very fast interviewers, whose removal changes nothing. The design limits what the answer
checks can say. With one interviewer per village, interviewer and village cannot be separated below the council.

## What the tests establish

The tests simulate surveys in which answers depend on area-level opinion and a respondent trait, so neighboring
respondents resemble each other and items are correlated. The fixture is [`tests/synthetic.py`](tests/synthetic.py).

- **Specificity.** On clean surveys, the share of per-interviewer p-values below 0.05 is close to 5%: 4.8% for
  effect heterogeneity (by permutation, which `auto` picks at about 60 respondents per arm) and 4.3% for balance (600 interviewer tests each), and 2.8% for straight-lining, 2.3% for
  pattern repetition and 5.6% for similarity (216 each). The two count-based checks come in under 5% because counts
  are discrete. The tests require each rate to stay below 10% and the p-values not to pile up at one. A full `screen`
  of a clean survey raises at most two single flags and no double flags. A check would fail these tests if it treated
  a neighbour's small-sample rate as known. An earlier draft of `straightlining` did that and flagged 10 of 60 clean
  interviewers.
- **Few respondents per arm.** With 30 interviewers, 4 respondents per arm and one common effect, Q against
  chi-square rejects in more than 40% of 60 simulated surveys and Bonferroni flags someone in more than 30%, while the
  permutation reference stays at or below 12% on both (about 5% over 1,000 runs). A test checks that every shuffle
  keeps each interviewer's number of treated and control units; letting the shuffle cross interviewers fails it, and
  fails the test that a planted interviewer-specific effect is still found. Another checks that the vectorized
  statistics on the observed labels equal the table's, in long format with respondent clusters.
- **Pakistan-style fabrication.** Two interviewers record a fixed string in about 60% of interviews, with different
  strings by arm, and mostly assign the invented ones to control. Both are flagged by `effect_heterogeneity` and by
  `pattern_repetition`. They are the two most lopsided interviewers in `assignment_balance`, and at least one survives
  correction there. Both are among the four most influential interviewers. No other interviewer is flagged by more
  than one check.
- **Whole-interview copying.** When copiers fill 40% of their interviews by copying their own earlier interviews with
  5% of answers changed, `interviewer_similarity` flags them and nobody else. Most interviews flagged by
  `percent_match` belong to the copiers, and in every such case the best partner has the same interviewer.
- **Clock.** In simulated paper schedules (40-minute interviews an hour apart, 2% honest slips of half an hour), an
  interviewer who logs a new interview every 25 minutes is flagged and nobody else is. Back-to-back interviews, where
  one ends the minute the next begins, raise nothing. A hand-built case checks that an interview overlapping an
  earlier, longer one is caught even when the interview just before it had ended.
- **Expected agreement.** With eight lopsided binary items, the raw 85% rule flags nearly everyone, while the
  expected-agreement test flags no one.

Power depends on how noisy genuine answers are. In these simulations a planted interviewer with 160 interviews is
caught by `effect_heterogeneity` in about 95% of runs when genuine answers bunch near the top of the scale, and in
about 70% when they spread across it. With 80 interviews the rates are lower. Pattern repetition has more power
against fixed strings, and balance has less power against moderate skew.

## Prior art: what exists and why this package

The searches covered CRAN, PyPI, SSC, and the survey-methods literature, in November 2026:

- **Stata `percentmatch`** (Kuriakose 2015, SSC S457984) computes the highest percent match between observations. It
  is a single statistic with no chance-agreement benchmark, no conditioning on area, and nothing at the interviewer
  level.
- **R `dataquieR`** (CRAN 2.8.15, September 2026; actively maintained) is a broad data-quality framework for
  epidemiological studies. Its `acc_varcomp` reports the share of variance due to an examiner or device (an ICC). It
  does not cover experiments, near duplicates, repeated answer strings, or nulls conditioned on area.
- **R `careless`** (CRAN 1.2.2, 2023) and its Python port **`careless-py`** target careless *respondents* (long-string
  indices, Mahalanobis distance). They work at the respondent level and have no interviewer or area structure.
- **PyPI `survey-dud-detector`** (0.2, 2020; inactive) checks individual responses for straight-lining and
  implausible combinations.
- No maintained package on CRAN or PyPI implements interviewer-level checks for survey experiments. That means
  effect heterogeneity by interviewer, leave-one-interviewer-out influence, or assignment balance by interviewer. No
  package implements a percent match judged against expected agreement either. The falsification literature below
  describes methods mostly as one-off analyses or unreleased code.

## Related work

Each strand below has the same logic as one or more checks here. Full entries, verified against Crossref or the
publisher, are in [`docs/references.bib`](docs/references.bib).

**Teacher cheating.** Jacob and Levitt (2003, *QJE* 118(3): 843–877) catch teachers who change their students'
answer sheets by combining two signals. The first is test-score gains in a classroom that are unexpectedly large and
then fade. The second is blocks of identical, suspicious answer strings shared within the classroom. The
interviewer-level analog is `effect_heterogeneity` (an unexpected interviewer-specific effect) paired with
`pattern_repetition` (the same answer string repeated within the interviewer's work). As in their design, either
signal alone is ambiguous and the two together are hard to explain innocently.

**Interviewer falsification in surveys.**

- AAPOR's 2003 statement on interviewer falsification and the AAPOR/ASA task force report *Falsification in Surveys*
  (DeMatteis et al. 2020) set out the practice standard of prevention, detection by data and process checks, and
  repair. Murphy et al. (2016, *Statistical Journal of the IAOS* 32(3): 313–326) review current practice. Murphy et
  al. (2004, *Proceedings of the ASA Survey Research Methods Section*, 4968–4975) describe a system for detecting
  falsification from interviewer-level indicators. `screen` is that idea reduced to statistics that need no paradata beyond start and end times.
- Schräpler and Wagner (2005, *Allgemeines Statistisches Archiv* 89(1): 7–20) study real faked interviews found in
  the SOEP raw data. Schräpler (2011, *Jahrbücher für Nationalökonomie und Statistik* 231(5–6): 685–718) uses
  Benford's-law digit tests to find them. Digit tests suit continuous answers such as incomes. This package works with categorical items, so it has no
  digit test.
- Bredl, Winker and Kötschau (2012, *Survey Methodology* 38(1): 1–10) and Menold et al. (2013, in *Interviewers'
  Deviations in Surveys*) cluster interviewers on indicators such as the share of extreme answers, of "other"
  responses and of non-response. Menold and Kemper (2014, *IJPOR* 26(1): 41–65) ground those indicators in the
  psychology of answering. Schwanhäuser, Sakshaug and Kosyakova (2022, *POQ* 86(1): 51–81) compare such methods
  against known falsifiers. `screen`'s per-interviewer table is the input such a multivariate step would use.
- Blasius and Thiessen (2012, *Assessing the Quality of Survey Data*; 2015, *Social Science Research* 52: 479–493)
  show that fabrication and simplified response behavior leave undifferentiated answers and duplicated response
  patterns. These correspond to `straightlining` and `pattern_repetition`.
- Timestamps are the oldest process check. Bushery, Reichert, Albright and Rossiter (1999, *Proceedings of the ASA
  Survey Research Methods Section*, 316–320) used the date and time stamps of Census Bureau CATI interviews to find
  interviewers who falsify. (That citation comes from the reference list of Schwanhäuser et al. 2022; the paper
  itself could not be retrieved.) Murphy et al. (2004) added question-level timings to NSDUH's detection system.
  Birnbaum, Borriello, Flaxman, DeRenzi and Karlin (2013, *Proceedings of CHI*, 2911–2920) classify fabricated
  interviews from the event logs of phone-based questionnaires. Schwanhäuser, Sakshaug and Kosyakova (2022) find
  interview duration among the most useful indicators against known falsifiers. `interview_clock` works with what
  paper schedules record, a start and end time per interview: overlaps that one person cannot produce, crowded days,
  and short interviews.
- Kuriakose and Robbins (2016, *Statistical Journal of the IAOS* 32(3): 283–291) propose the maximum percent match
  with an 85% threshold. Simmons, Mercer, Schwarzer and Kennedy (2016, same issue, 327–338) show that high matches
  arise in valid data and depend on survey design. `percent_match` implements the statistic and replaces the
  threshold with a chance-agreement benchmark.

**Election forensics.** Beber and Scacco (2012, *Political Analysis* 20(2): 211–234) test whether the last digits of
vote counts are uniform, because people who invent numbers pick digits non-randomly. Mebane (2008, in *Election
Fraud*, Brookings) applies second-digit Benford tests to US presidential returns. The shared logic is to compare the
data's fine structure with what an unmanipulated process produces. Here that comparison is `pattern_repetition`
against the shuffled-label distribution. Digit tests themselves are not implemented, because survey answers are
categorical.

**Fabricated scientific data.**

- Broockman, Kalla and Aronow (2015, "Irregularities in LaCour (2014)") found that the panel responses in LaCour
  and Green resembled an existing survey plus regular noise, with too little of the irregularity real respondents
  show. The analogs here are `percent_match` and `interviewer_similarity` (responses too close to a source) and the
  zero-variance flag (answers too regular).
- Simonsohn (2013, *Psychological Science* 24(10): 1875–1888) detects fabrication from summary statistics that are
  too similar across conditions to have come from sampling. Brown and Heathers (2017, *Social Psychological and
  Personality Science* 8(4): 363–369) check whether reported means are possible given the sample size (the GRIM
  test). Both measure variance that is too small. The zero-variance flag in `effect_heterogeneity` and the
  too-close-twins logic of `interviewer_similarity` measure the same thing.
- Carlisle (2017, *Anaesthesia* 72(8): 944–952) tests baseline balance across thousands of randomized trials, on the
  grounds that fabricated trials are too balanced or not balanced enough. That is `assignment_balance` at the
  interviewer level.

**The case studies.** Blair, Fair, Malhotra and Shapiro (2013), "Poverty and Support for Militant Politics: Evidence
from Pakistan," *American Journal of Political Science* 57(1): 30–48. Beaman, Duflo, Pande and Topalova (2012),
"Female Leadership Raises Aspirations and Educational Attainment for Girls: A Policy Experiment in India,"
*Science* 335(6068): 582–586.

## Development

```bash
make install    # uv sync
make format     # isort + black
make lint       # black --check, isort --check-only, flake8 (line length 120)
make test       # pytest
make check      # lint + test
make ci-docker  # make check inside the standard python:3.13 image
```

The version lives only in `pyproject.toml`, and `interviewer_checks.__version__` reads it from the installed metadata.

## License

MIT
