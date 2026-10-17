"""Single-file HTML report: demo experiment + validation evidence.

Charts are rendered with matplotlib (Agg), embedded as base64 PNGs, so the
report is one self-contained file with no JavaScript and no network calls.

Validation uses the project's one standardized Monte Carlo configuration so
the README, CLI defaults, and this report always agree:
  binomial model:  20,000 null reps / 10,000 alt reps, 20 looks, seed 12345
  simulator model:  2,000 null reps /  1,000 alt reps, 4 weekly looks, seed 777
"""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .simulator import CampaignSimulator, CreativeSpec
from .experiment import obf_bounds, z_test_two_proportions
from .validation import run_validation, validate_via_simulator

# Standardized validation configuration (keep in sync with README + CLI).
MC_REPS_NULL = 20_000
MC_REPS_ALT = 10_000
MC_SEED = 12345
SIM_REPS_NULL = 2_000
SIM_REPS_ALT = 1_000
SIM_SEED = 777


def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_demo(seed: int = 7):
    """Run the demo scenario: two creatives, 28 days, weekly sequential looks."""
    sim = CampaignSimulator([
        CreativeSpec(name="Control (A)", true_ctr=0.020, true_cvr=0.100,
                     fatigue_rate=0.03, daily_budget=500),
        CreativeSpec(name="Challenger (B)", true_ctr=0.022, true_cvr=0.115,
                     fatigue_rate=0.02, daily_budget=500),
    ], seed=seed)
    rows = sim.run(28)

    days = sorted({r.day for r in rows})
    per = {c: {"imp": [], "clk": [], "conv": []} for c in
           ["Control (A)", "Challenger (B)"]}
    for d in days:
        for r in rows:
            if r.day == d:
                per[r.creative]["imp"].append(r.impressions)
                per[r.creative]["clk"].append(r.clicks)
                per[r.creative]["conv"].append(r.conversions)

    # Weekly looks on CVR (per click), equal information spacing.
    looks = []
    for wk, frac in zip([7, 14, 21, 28], [0.25, 0.50, 0.75, 1.0]):
        sl = slice(0, wk)
        x_t = sum(per["Challenger (B)"]["conv"][sl]); n_t = sum(per["Challenger (B)"]["clk"][sl])
        x_c = sum(per["Control (A)"]["conv"][sl]); n_c = sum(per["Control (A)"]["clk"][sl])
        looks.append((frac, x_t, n_t, x_c, n_c))
    bounds = obf_bounds([l[0] for l in looks])
    zpath = []
    for (frac, x_t, n_t, x_c, n_c), b in zip(looks, bounds):
        z, p = z_test_two_proportions(x_t, n_t, x_c, n_c)
        zpath.append((frac, z, b, p))

    return rows, per, days, zpath


def render_report(out_path: str, seed: int = 7) -> dict:
    rows, per, days, zpath = build_demo(seed=seed)
    null, alt = run_validation(reps_null=MC_REPS_NULL, reps_alt=MC_REPS_ALT,
                               seed=MC_SEED)
    simval = validate_via_simulator(reps_null=SIM_REPS_NULL,
                                    reps_alt=SIM_REPS_ALT, seed=SIM_SEED)

    # Chart 1: CTR trajectories with fatigue.
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for name, color in [("Control (A)", "#4878cf"), ("Challenger (B)", "#d65f5f")]:
        ctr = [c / i for c, i in zip(per[name]["clk"], per[name]["imp"])]
        ax.plot(days, ctr, marker="o", ms=3, lw=1.2, label=name, color=color)
    ax.set_title("Observed daily CTR (fatigue drags both down; B starts higher)")
    ax.set_xlabel("day"); ax.set_ylabel("CTR"); ax.legend()
    chart_ctr = _fig_b64(fig)

    # Chart 2: cumulative conversions.
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for name, color in [("Control (A)", "#4878cf"), ("Challenger (B)", "#d65f5f")]:
        ax.plot(days, np.cumsum(per[name]["conv"]), lw=1.6, label=name, color=color)
    ax.set_title("Cumulative conversions (two effects combined - see estimand note)")
    ax.set_xlabel("day"); ax.set_ylabel("conversions"); ax.legend()
    chart_conv = _fig_b64(fig)

    # Chart 3: z path vs OBF bounds at weekly looks.
    fig, ax = plt.subplots(figsize=(7, 3.4))
    fracs = [z[0] for z in zpath]; zs = [z[1] for z in zpath]; bs = [z[2] for z in zpath]
    ax.plot(fracs, zs, marker="o", color="#333", label="cumulative z at each weekly look")
    ax.step(fracs, bs, where="post", color="#d65f5f", label="O'Brien-Fleming boundary")
    ax.step(fracs, [-b for b in bs], where="post", color="#d65f5f")
    ax.axhline(1.96, ls="--", lw=1, color="#4878cf", label="naive fixed-horizon 1.96")
    ax.set_title("Sequential monitoring of the CVR experiment (weekly looks)")
    ax.set_xlabel("information fraction"); ax.set_ylabel("z"); ax.legend(fontsize=8)
    chart_seq = _fig_b64(fig)

    # Chart 4: validation summary (binomial model, standardized config).
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = ["Naive daily peeking\n(false +ve rate)", "OBF boundaries\n(false +ve rate)",
              "OBF boundaries\n(power at +1.5pt CVR)"]
    vals = [null.naive_peeking_fpr, null.obf_fpr, alt.obf_power]
    colors = ["#d65f5f", "#6acc65", "#4878cf"]
    bars = ax.bar(labels, vals, color=colors)
    ax.axhline(0.05, ls="--", lw=1, color="#888")
    ax.text(-0.55, 0.062, "nominal 5%", fontsize=8, color="#555")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center", fontsize=10)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, max(vals) * 1.25); ax.set_ylabel("rate")
    ax.set_title(f"Binomial-model Monte Carlo ({null.reps:,} null reps, {alt.reps:,} alt reps)")
    chart_val = _fig_b64(fig)

    # First crossing drives the demo narrative; later looks are counterfactual.
    final = zpath[-1]
    first_cross = next((i + 1 for i, (f, z, b, p) in enumerate(zpath) if abs(z) > b), None)
    if first_cross is not None:
        verdict = (f"first crossed the O&#39;Brien-Fleming boundary at look {first_cross} "
                   f"of {len(zpath)} ({zpath[first_cross-1][0]:.0%} of planned information; "
                   f"z&nbsp;=&nbsp;{zpath[first_cross-1][1]:+.2f} vs boundary {zpath[first_cross-1][2]:.2f})")
    else:
        verdict = (f"did not cross the O&#39;Brien-Fleming boundary at any look "
                   f"(final z&nbsp;=&nbsp;{final[1]:+.2f} vs boundary {final[2]:.2f})")

    def decision_cell(i, f, z, b):
        if first_cross is not None and i + 1 > first_cross:
            return "counterfactual - experiment already stopped"
        return "STOP: significant" if abs(z) > b else "continue"

    table_rows = "".join(
        f"<tr><td>{i+1}</td><td>{f:.0%}</td><td>{z:+.2f}</td><td>{b:.2f}</td>"
        f"<td>{decision_cell(i, f, z, b)}</td></tr>"
        for i, (f, z, b, p) in enumerate(zpath))

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>campaign_sim demo report</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 860px;
       margin: 2em auto; color: #222; line-height: 1.5; padding: 0 1em; }}
img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; margin: 0.6em 0 1.4em; }}
table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #ccc; padding: 4px 10px; }}
h1 {{ font-size: 1.5em; }} h2 {{ font-size: 1.15em; margin-top: 1.6em; }}
.small {{ color: #666; font-size: 0.85em; }}
</style></head><body>
<h1>campaign_sim &mdash; demo report</h1>
<p class="small">All data on this page is synthetic, generated by the bundled
CampaignSimulator with known ground truth. No ad platform, account, or real
campaign data was touched.</p>

<h2>Scenario</h2>
<p>Two creatives, equal $500/day budgets, 28 days. Ground truth: Control A
converts 10.0% of clicks with 2.0% base CTR; Challenger B converts 11.5%
(+1.5 points absolute) with 2.2% base CTR. CTR decays with fatigue;
impressions saturate against a finite audience; day-of-week seasonality
applies. The question an operator asks before spending: <em>is B's conversion
lift real, and when can we call it?</em></p>
<img src="data:image/png;base64,{chart_ctr}" alt="daily CTR">
<img src="data:image/png;base64,{chart_conv}" alt="cumulative conversions">
<p><strong>Estimand.</strong> The sequential test below compares
<strong>CVR - conversions per click</strong> - between the creatives. Because
the Challenger also has a higher CTR, differences in <em>total</em>
conversions, spend efficiency, or CPA combine both effects and are not a clean
causal read on any single one; the cumulative-conversions chart is descriptive,
not the test. A production harness would pre-register the primary metric and
randomize budget allocation.</p>

<h2>Sequential analysis (weekly looks)</h2>
<p>Checking an unadjusted z-test every day and stopping at p&nbsp;&lt;&nbsp;0.05
feels rigorous and is not: across {null.reps:,} simulated experiments with
<strong>no true effect</strong>, that rule produced a winner
{null.naive_peeking_fpr:.1%} of the time &mdash; roughly five times the promised
5%. O'Brien-Fleming spending-function boundaries start extremely strict and
relax as information accrues, keeping the overall false-positive rate at
{null.obf_fpr:.1%} (nominal 5%).</p>
<img src="data:image/png;base64,{chart_seq}" alt="sequential z path">
<table>
<tr><th>look</th><th>information</th><th>z</th><th>boundary</th><th>decision</th></tr>
{table_rows}
</table>
<p>In this run the Challenger {verdict}; rows after that look are the
counterfactual path had the experiment continued. The naive 1.96 rule would
have called it at the first look, when a false call is most likely.</p>

<h2>Validation evidence</h2>
<p>Two separate validation models, each labeled. Both use fixed seeds and the
standardized configuration below, so reproducing the commands reproduces these
exact numbers.</p>
<img src="data:image/png;base64,{chart_val}" alt="validation">
<h3>Model 1: binomial Monte Carlo (<code>python -m campaign_sim validate</code>)</h3>
<p class="small">{null.reps:,} null reps / {alt.reps:,} alt reps, 20 daily looks,
seed {MC_SEED}, control rate 0.10, MDE +0.015. Cumulative counts within each
simulated experiment are built by cumulative summation, so z-statistics across
looks are correlated exactly as in a real sequentially monitored experiment.</p>
<table>
<tr><th>check</th><th>result</th><th>expectation</th></tr>
<tr><td>naive daily peeking, null worlds</td><td>{null.naive_peeking_fpr:.2%} flagged</td><td>far above 5% (inflation demonstrated)</td></tr>
<tr><td>OBF boundaries, null worlds</td><td>{null.obf_fpr:.2%} flagged</td><td>&asymp; 5% nominal</td></tr>
<tr><td>OBF power, +1.5pt worlds</td><td>{alt.obf_power:.2%}</td><td>near the 80% fixed-horizon design target</td></tr>
<tr><td>looks used when significant</td><td>{alt.obf_mean_looks_used:.1f} / {alt.looks} avg</td><td>early stops save sample</td></tr>
</table>
<h3>Model 2: full simulator pipeline (<code>python -m campaign_sim validate-sim</code>)</h3>
<p class="small">{simval['reps_null']:,} null reps / {simval['reps_alt']:,} alt reps,
{simval['days']} days, {simval['looks']} weekly looks, seed {SIM_SEED}. Each rep runs the
complete CampaignSimulator (Poisson impressions, saturation, fatigue,
seasonality) and pipes its daily rows through the same weekly sequential
analysis as the demo above. Null worlds use two identical creatives; alt
worlds differ only in Challenger CVR (+0.015).</p>
<table>
<tr><th>check</th><th>result</th><th>expectation</th></tr>
<tr><td>naive weekly peeking, null worlds</td><td>{simval['null_naive_fpr']:.2%} flagged</td><td>above 5% (inflation, smaller with only {simval['looks']} looks)</td></tr>
<tr><td>OBF boundaries, null worlds</td><td>{simval['null_obf_fpr']:.2%} flagged</td><td>&asymp; 5% nominal</td></tr>
<tr><td>OBF power, +1.5pt CVR worlds</td><td>{simval['alt_obf_power']:.2%}</td><td>high at this sample volume</td></tr>
<tr><td>looks used when significant</td><td>{simval['alt_obf_mean_looks_used']:.1f} / {simval['looks']} avg</td><td>early stops save time</td></tr>
</table>

<h2>Honest limitations</h2>
<ul>
<li>The world is synthetic by construction; fidelity to any real account is
    unvalidated. It is a decision-support sandbox, not a forecast.</li>
<li>Normal-approximation z-tests need adequate counts per arm; very low
    conversion volumes break the approximation.</li>
<li>Group-sequential monitoring costs a little power at equal total N
    ({alt.obf_power:.1%} measured vs the 80% fixed-horizon target); production
    designs should inflate sample size slightly to compensate.</li>
<li>Weekly looks in the demo and Model 2 assume equal information spacing;
    unequal spacing is supported by the API but not exercised here.</li>
<li>No ad-platform integration exists or is claimed. Inputs are simulator
    output or user-supplied CSVs.</li>
</ul>
<p class="small">Generated by campaign_sim. Reproduce: <code>python -m campaign_sim report --out report.html</code></p>
</body></html>"""
    with open(out_path, "w") as fh:
        fh.write(html)
    return {"null": null.to_dict(), "alt": alt.to_dict(), "sim": simval,
            "demo_final_z": final[1], "demo_final_bound": final[2],
            "out": out_path}
