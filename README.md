# interviewer-checks

[![PyPI version](https://img.shields.io/pypi/v/interviewer-checks.svg)](https://pypi.org/project/interviewer-checks/)
[![CI](https://github.com/finite-sample/interviewer-checks/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/interviewer-checks/actions/workflows/ci.yml)

Find data problems tied to interviewers in face-to-face surveys and survey experiments.

An interviewer who invents interviews, copies them, or skips the randomization leaves traces. Three are common:
estimates that depend on a handful of interviewers, treatment effects that differ by interviewer more than sampling
noise allows, and answers that repeat within an interviewer's work more than local opinion explains. The package
looks for those generic traces rather than for any particular pattern. You don't need to know what the problem looks
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
| `effect_heterogeneity(df, outcome, treatment, cluster, respondent=None)` | each interviewer's treated-minus-control difference against the pooled one, with respondent-clustered SEs in long format | Cochran's Q across interviewers; per-interviewer z, with Bonferroni | interviewer-specific treatment effects; interviewers with zero standard error |
| `assignment_balance(df, treatment, cluster)` | each interviewer's treated count against the overall share | exact binomial, with Bonferroni; chi-square of interviewer × arm | interviews that skipped the randomization |
| `percent_match(df, items, area=None)` | each interview's best match (share of identical answers) against the agreement expected by chance from item distributions in its area | items agree independently with those chances; Bonferroni over all pairs searched | copied or duplicated interviews |
| `interviewer_similarity(df, items, interviewer, area)` | how close each interview's nearest twin among the same interviewer's work is, against shuffled interviewer labels | shuffle interviewer labels within area | interviewers whose interviews resemble each other |
| `pattern_repetition(df, items, interviewer, area, strata=None)` | how many of an interviewer's interviews carry their most common full answer string, by stratum (e.g. arm) | shuffle interviewer labels within area × stratum | interviewers who write the same answers over and over |
| `straightlining(df, items, interviewer, area)` | each interviewer's count of interviews that give one answer to every item | exact: a sum over areas of hypergeometric counts (the shuffle null in closed form) | interviewers who take the shortcut of one answer to everything |
| `screen(df, interviewer, ...)` | runs whichever checks the columns allow | | one table with `flag_*` columns, plus a text summary |

Each docstring spells out the mechanism, so `help(interviewer_checks.percent_match)` says exactly what is compared.

### Two design choices that matter

**Nulls condition on area.** People who live near each other answer alike. If interviewers work in different areas,
"interviewer X's answers are unusual" may just mean that X works somewhere unusual. So every interviewer-level null
shuffles interviewer labels *within* an area and keeps how many interviews each interviewer did there. That way an
interviewer is compared only with neighbours who interviewed the same population.

The weak point is that an area worked by a single interviewer can't be shuffled. There, interviewer and area are the
same thing, and no data-only check can tell them apart. Each check reports `single_interviewer_areas` and
`single_interviewer_interviews` in `.attrs` and leaves those interviews out. If most of your sampling points have one
interviewer (common: one interviewer per cluster is a standard field design), pass a coarser area such as the district.
Then remember that the comparison also absorbs differences between villages inside the district. The case study below
shows what that costs.

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
Pooled effect -0.028 (SE 0.004; 0.005 clustered by interviewer). Heterogeneity across interviewers: chi2 = 99.6 on 59 df, p = 0.000746. Interviewer effects off the pooled one: 12, 22. No variation in the outcome: none.
Leaving one interviewer out moves the effect most for 12 (-0.0043, -0.8 jackknife SE), 22 (-0.0013, -0.2 jackknife SE), 14 (+0.0010, +0.2 jackknife SE).
Assignment: 49.5% treated overall; interviewer x treatment chi2 = 68.4 on 59 df, p = 0.188. Lopsided interviewers: none.
Repeated answer strings beyond what the area produces: 12.
Straight-lining (9.3% of interviews overall) above the area rate: none.
Flagged by two or more checks: 12.
  interviewer  effect_z  share_treated  repeated_share               flags
0          12  6.140406       0.350000        0.583333  effect, repetition
1          22  3.357077       0.450000        0.075000              effect
2           0  1.873818       0.508333        0.075000
3           1 -1.917643       0.508333        0.075000
```

The planted interviewer is the only one flagged twice. Interviewer 22 is a false positive on one check: Bonferroni
holds the chance of any false flag to about 5% per check, not zero. Interviewer 12's 35% treated share over 120
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
5,479 respondents, 75 interviewers

## effect_heterogeneity (long format, SEs clustered on respondent)
pooled ATE -0.013 (SE 0.005); Cochran's Q = 93.1 on 59 df, p = 0.0031; zero standard error: [33, 35]
 interviewer  respondents  treated  control  effect    se     z  p_adjusted  flagged
          33           12        4        8   0.000 0.000   inf         NaN    False
          35           12        4        8   0.000 0.000   inf         NaN    False
          71           52       17       35   0.047 0.014 4.445   5.275e-04     True
          37          160       60      100   0.082 0.024 3.998   3.829e-03     True
           7          101       48       53   0.037 0.021 2.344   1.000e+00    False
without 7, 37, 71: ATE -0.021; Q = 54.8 on 56 df, p = 0.519

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
Pooled effect -0.013 (SE 0.005; 0.007 clustered by interviewer). Heterogeneity across interviewers: chi2 = 118.0 on 59 df, p = 7.98e-06. Interviewer effects off the pooled one: 37, 71. No variation in the outcome: 33, 35.
Leaving one interviewer out moves the effect most for 37 (-0.0047, -0.7 jackknife SE), 66 (+0.0022, +0.3 jackknife SE), 71 (-0.0018, -0.3 jackknife SE).
Assignment: 49.2% treated overall; interviewer x treatment chi2 = 76.3 on 74 df, p = 0.404. Lopsided interviewers: none.
Repeated answer strings beyond what the area produces: 7, 10, 12, 22, 24, 29, 37, 41, 44, 45, 55, 56, 61, 63, 66, 71.
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
  allows (Q = 93.1 on 59 df, p = 0.003). The two most extreme, 71 and 37, survive a Bonferroni correction over 60
  interviewers. Interviewer 7 is next (z = 2.3), but that is not significant once you correct for looking at 60
  interviewers. Without 7, 37 and 71, the pooled effect moves from −0.013 to −0.021 and the heterogeneity disappears
  (Q = 54.8 on 56 df, p = 0.52). Interviewers 33 and 35 have zero standard error: every respondent's answers equal
  their arm's mean, so neither interviewer has any respondent-to-respondent variation. (The wide-format run in
  `screen` drops the 35 respondents who got both endorsed and unendorsed items. Without them, 71's standard error
  shrinks and Q rises to 118.)
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

## What the tests establish

The tests simulate surveys in which answers depend on area-level opinion and a respondent trait, so neighbouring
respondents resemble each other and items are correlated. The fixture is [`tests/synthetic.py`](tests/synthetic.py).

- **Specificity.** On clean surveys, the share of per-interviewer p-values below 0.05 is close to 5%: 5.0% for
  effect heterogeneity and 4.3% for balance (600 interviewer tests each), and 2.8% for straight-lining, 2.3% for
  pattern repetition and 5.6% for similarity (216 each). The two count-based checks come in under 5% because counts
  are discrete. The tests require each rate to stay below 10% and the p-values not to pile up at one. A full `screen`
  of a clean survey raises at most two single flags and no double flags. A check would fail these tests if it treated
  a neighbour's small-sample rate as known. An earlier draft of `straightlining` did that and flagged 10 of 60 clean
  interviewers.
- **Pakistan-style fabrication.** Two interviewers record a fixed string in about 60% of interviews, with different
  strings by arm, and mostly assign the invented ones to control. Both are flagged by `effect_heterogeneity` and by
  `pattern_repetition`. They are the two most lopsided interviewers in `assignment_balance`, and at least one survives
  correction there. Both are among the four most influential interviewers. No other interviewer is flagged by more
  than one check.
- **Whole-interview copying.** When copiers fill 40% of their interviews by copying their own earlier interviews with
  5% of answers changed, `interviewer_similarity` flags them and nobody else. Most interviews flagged by
  `percent_match` belong to the copiers, and in every such case the best partner has the same interviewer.
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
  falsification from interviewer-level indicators. `screen` is that idea reduced to statistics that don't need paradata.
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
  show that fabrication and simplified response behaviour leave undifferentiated answers and duplicated response
  patterns. These correspond to `straightlining` and `pattern_repetition`.
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

**The case study.** Blair, Fair, Malhotra and Shapiro (2013), "Poverty and Support for Militant Politics: Evidence
from Pakistan," *American Journal of Political Science* 57(1): 30–48.

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
