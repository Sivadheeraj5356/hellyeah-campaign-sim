# hellyeah-campaign-sim

A synthetic paid-media **campaign simulator** and **experimentation harness**:
model creative performance before spending real money, and analyze A/B tests
on the simulated data without inflating false positives.

Built as a work sample for the AI Engineer, Learn Engine: Platform & Simulation
role at Hellyeah AI — the role owns "the campaign simulator, experimentation
infrastructure... that let operators model outcomes before spending real
money." This is a small, honest slice of that surface: a simulator with known
ground truth, plus the statistics to make decisions on it safely, with every
statistical claim measured by Monte Carlo validation rather than asserted.

Everything here is synthetic. No ad platform, account, or real campaign data
is touched, and no integration with one exists or is claimed.

## What it does

- **Simulator** (`campaign_sim/simulator.py`) — daily impressions/clicks/
  conversions/spend per creative with budget saturation (diminishing returns),
  creative fatigue (CTR decay), day-of-week seasonality, and binomial noise.
  Ground truth is explicit, so experiments have a known correct answer.
- **Fixed-horizon stats** (`campaign_sim/experiment.py`) — two-proportion
  z-test and per-arm sample-size planning for a target MDE.
- **Sequential monitoring** — O'Brien-Fleming-style group-sequential
  boundaries computed from the Lan-DeMets alpha-spending function via
  grid-based recursive integration (Armitage-McPherson-Rowe). No lookup
  tables; a single look reduces exactly to the fixed-horizon 1.96.
- **Validation, two labeled models** (`campaign_sim/validation.py`) —
  (1) a vectorized binomial Monte Carlo whose per-experiment z-statistics are
  built by cumulative summation, so looks are correlated exactly as in a real
  sequentially monitored experiment; (2) an end-to-end pipeline validation
  that runs the full CampaignSimulator per rep and pipes its daily rows
  through the same weekly sequential analysis as the demo.
- **Report** (`campaign_sim/report.py`) — single-file HTML demo report
  (matplotlib charts embedded as base64; no JS, no network).

## Standardized validation configuration

One fixed configuration is used everywhere — CLI defaults, the HTML report,
and this README — so reproducing the commands reproduces these exact numbers.
Both models use fixed seeds.

**Model 1: binomial Monte Carlo** — `python -m campaign_sim validate`
(20,000 null reps / 10,000 alt reps, 20 daily looks, seed 12345, control rate
0.10, MDE +0.015, alpha 0.05):

| decision rule | rate | reference |
|---|---|---|
| naive daily peeking, stop at p < 0.05 | **24.775%** false positives | nominal 5% |
| OBF spending-function boundaries | **4.975%** false positives | nominal 5% |
| OBF power at true +1.5pt lift | **77.410%** | 80% fixed-horizon design target |
| looks used when significant | **14.7 / 20** avg | early stops save sample |

**Model 2: full simulator pipeline** — `python -m campaign_sim validate-sim`
(2,000 null reps / 1,000 alt reps, 28 days, 4 weekly looks, seed 777; null
worlds use two identical creatives, alt worlds differ only in CVR +0.015):

| decision rule | rate | reference |
|---|---|---|
| naive weekly peeking | **11.850%** false positives | nominal 5% (inflation is smaller with only 4 looks) |
| OBF boundaries | **4.550%** false positives | nominal 5% |
| OBF power at true +1.5pt CVR lift | **99.000%** | high at this sample volume |
| looks used when significant | **2.3 / 4** avg | early stops save time |

Checking an unadjusted z-test on every new day of data and stopping at
p < 0.05 feels rigorous and is not: Model 1 shows a ~5x false-positive
inflation. The spending-function boundaries start extremely strict and relax
as information accrues, holding the overall rate at alpha in both models.

## Quickstart

```bash
pip install numpy scipy matplotlib   # only deps

python -m campaign_sim simulate --days 28 --out simulation.csv
python -m campaign_sim bounds --looks 5
python -m campaign_sim validate            # Model 1, standard config
python -m campaign_sim validate-sim        # Model 2, standard config
python -m campaign_sim report --out report.html
python -m unittest discover -s tests       # 22 tests
```

`bounds --looks 5` (equal information spacing, alpha = 0.05):

```
look  1  info  20.0%  boundary 4.3827
look  2  info  40.0%  boundary 3.0990
look  3  info  60.0%  boundary 2.5562
look  4  info  80.0%  boundary 2.2528
look  5  info 100.0%  boundary 2.0650
```

## Estimand note for the demo

The demo's sequential test compares **CVR (conversions per click)** between
two creatives. The creatives also differ in CTR, so differences in total
conversions or CPA combine both effects and are descriptive, not a clean
causal read on any single metric. A production harness would pre-register the
primary metric and randomize budget allocation.

## Honest limitations

- The world is synthetic by construction; fidelity to any real ad account is
  unvalidated. This is a decision-support sandbox, not a forecast.
- Normal-approximation z-tests need adequate counts per arm; very low
  conversion volumes break the approximation.
- Sequential monitoring trades a little power for the right to look early
  (77.4% measured vs the 80% fixed-horizon target at equal N in Model 1);
  production designs should inflate sample size slightly to compensate.
- The Lan-DeMets spending function approximates exact O'Brien-Fleming
  boundaries (classic K=5 constants differ slightly); calibration here is
  established by the Monte Carlo FPR measurements above, not by table-matching.
- Unequal information spacing is supported by the API (`obf_bounds` takes
  arbitrary increasing fractions) but the demos use equal spacing.
- No ad-platform integration, no user data, no company internals.

## Layout

```
campaign_sim/
  simulator.py    synthetic campaign world (saturation, fatigue, seasonality)
  experiment.py   z-test, sample size, alpha spending, OBF boundaries
  validation.py   binomial MC + full-simulator pipeline MC
  report.py       single-file HTML demo report
  cli.py          simulate / bounds / validate / validate-sim / report
tests/            22 unit tests
```

MIT licensed. Original code.
