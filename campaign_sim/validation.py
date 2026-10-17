"""Monte Carlo validation of the experiment methods.

Every statistical claim this project makes is measured, not asserted:

- naive_peeking_fpr: simulate experiments with NO true effect, check an
  unadjusted z-test every day, stop at the first p < alpha. The fraction of
  simulated experiments that ever "find significance" is the real false
  positive rate of daily peeking.
- obf_fpr: same null worlds, but decisions use the O'Brien-Fleming
  spending-function boundaries. Should land near alpha.
- obf_power: worlds WITH a true effect; fraction correctly rejected is the
  empirical power, comparable to the fixed-horizon design target.

Vectorized with numpy; a few thousand reps take seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np
from scipy.stats import norm

from .experiment import obf_bounds, sample_size_per_arm


@dataclass
class ValidationResult:
    scenario: str
    reps: int
    looks: int
    n_per_arm_per_look: int
    p_control: float
    p_treatment: float
    alpha: float
    naive_peeking_fpr: float | None
    obf_fpr: float | None
    obf_power: float | None
    obf_mean_looks_used: float | None

    def to_dict(self):
        return asdict(self)


def _simulate_matrix(rng, reps, looks, n_per_look, p_c, p_t):
    """Cumulative conversions and clicks per rep/look. Returns (xc, nc, xt, nt)."""
    conv_c = rng.binomial(n_per_look, p_c, size=(reps, looks)).cumsum(axis=1)
    conv_t = rng.binomial(n_per_look, p_t, size=(reps, looks)).cumsum(axis=1)
    n = (np.arange(1, looks + 1) * n_per_look)[None, :]
    return conv_c, np.broadcast_to(n, (reps, looks)), conv_t, np.broadcast_to(n, (reps, looks))


def _z_matrix(xc, nc, xt, nt):
    p_pool = (xc + xt) / (nc + nt)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / nc + 1 / nt))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (xt / nt - xc / nc) / se, 0.0)
    return z


def run_validation(p_control: float = 0.10, mde_abs: float = 0.015,
                   alpha: float = 0.05, looks: int = 20,
                   n_per_arm_per_look: int | None = None,
                   reps_null: int = 20000, reps_alt: int = 10000,
                   seed: int = 12345) -> tuple[ValidationResult, ValidationResult]:
    """Null-world FPR validation + alternative-world power validation.

    If n_per_arm_per_look is None, sizes each look so the total per-arm
    sample equals the fixed-horizon design for (p_control, mde_abs, alpha,
    power=0.80) split evenly across `looks` looks.
    """
    if n_per_arm_per_look is None:
        total_n = sample_size_per_arm(p_control, mde_abs, alpha=alpha, power=0.80)
        n_per_arm_per_look = max(1, math.ceil(total_n / looks))
    fracs = [(i + 1) / looks for i in range(looks)]
    bounds = np.array(obf_bounds(fracs, alpha=alpha))
    z_alpha = norm.ppf(1 - alpha / 2)
    rng = np.random.default_rng(seed)

    # --- Null worlds: no true effect. ---
    xc, nc, xt, nt = _simulate_matrix(rng, reps_null, looks, n_per_arm_per_look,
                                      p_control, p_control)
    z = _z_matrix(xc, nc, xt, nt)
    naive_rej = (np.abs(z) > z_alpha).any(axis=1).mean()
    obf_rej = (np.abs(z) > bounds[None, :]).any(axis=1).mean()

    null = ValidationResult(
        scenario="null (no true effect)", reps=reps_null, looks=looks,
        n_per_arm_per_look=n_per_arm_per_look, p_control=p_control,
        p_treatment=p_control, alpha=alpha,
        naive_peeking_fpr=float(naive_rej), obf_fpr=float(obf_rej),
        obf_power=None, obf_mean_looks_used=None)

    # --- Alternative worlds: true lift of mde_abs. ---
    xc, nc, xt, nt = _simulate_matrix(rng, reps_alt, looks, n_per_arm_per_look,
                                      p_control, p_control + mde_abs)
    z = _z_matrix(xc, nc, xt, nt)
    crossed = np.abs(z) > bounds[None, :]
    power = crossed.any(axis=1).mean()
    first_cross = np.where(crossed.any(axis=1), crossed.argmax(axis=1) + 1, looks)
    alt = ValidationResult(
        scenario=f"alternative (+{mde_abs:.4f} absolute)", reps=reps_alt,
        looks=looks, n_per_arm_per_look=n_per_arm_per_look, p_control=p_control,
        p_treatment=p_control + mde_abs, alpha=alpha,
        naive_peeking_fpr=None, obf_fpr=None, obf_power=float(power),
        obf_mean_looks_used=float(first_cross.mean()))
    return null, alt


def validate_via_simulator(days: int = 28, looks: int = 4,
                           p_cvr_control: float = 0.10, mde_abs: float = 0.015,
                           alpha: float = 0.05,
                           reps_null: int = 2000, reps_alt: int = 1000,
                           seed: int = 777) -> dict:
    """End-to-end validation on actual CampaignSimulator output.

    Separate model from run_validation: instead of drawing binomial counts
    directly, each rep runs the full simulator (fatigue, seasonality,
    saturation, Poisson impressions) for two creatives and pipes the resulting
    daily rows through the same weekly-look sequential analysis the demo
    report uses. This checks the whole pipeline on data with realistic
    structure, not only the idealized Brownian/binomial model.

    Null worlds: both creatives identical (same CTR, CVR, fatigue, budget).
    Alt worlds: identical except Challenger CVR = control + mde_abs.
    Looks: weekly, equal information spacing (documented approximation).
    """
    from .simulator import CampaignSimulator, CreativeSpec
    from .experiment import obf_bounds, z_test_two_proportions

    fracs = [(i + 1) / looks for i in range(looks)]
    bounds = obf_bounds(fracs, alpha=alpha)
    z_alpha = norm.ppf(1 - alpha / 2)
    days_per_look = days // looks

    def one_world(rep_seed: int, effect: float) -> list[float]:
        sim = CampaignSimulator([
            CreativeSpec("A", 0.020, p_cvr_control, fatigue_rate=0.03, daily_budget=500),
            CreativeSpec("B", 0.020, p_cvr_control + effect, fatigue_rate=0.03, daily_budget=500),
        ], seed=rep_seed)
        rows = sim.run(days)
        zs = []
        for lk in range(1, looks + 1):
            upto = lk * days_per_look
            xc = sum(r.conversions for r in rows if r.creative == "A" and r.day < upto)
            nc = sum(r.clicks for r in rows if r.creative == "A" and r.day < upto)
            xt = sum(r.conversions for r in rows if r.creative == "B" and r.day < upto)
            nt = sum(r.clicks for r in rows if r.creative == "B" and r.day < upto)
            z, _ = z_test_two_proportions(xt, nt, xc, nc)
            zs.append(z)
        return zs

    def sweep(reps: int, effect: float, seed0: int):
        naive = obf = 0
        looks_used = []
        for r in range(reps):
            zs = one_world(seed0 + r, effect)
            if any(abs(z) > z_alpha for z in zs):
                naive += 1
            crossed = next((i for i, z in enumerate(zs) if abs(z) > bounds[i]), None)
            if crossed is not None:
                obf += 1
                looks_used.append(crossed + 1)
        return naive / reps, obf / reps, (float(np.mean(looks_used)) if looks_used else None)

    n_naive, n_obf, _ = sweep(reps_null, 0.0, seed)
    a_naive, a_obf, a_looks = sweep(reps_alt, mde_abs, seed + 10_000_000)
    return {
        "model": "full CampaignSimulator pipeline (fatigue + seasonality + saturation)",
        "days": days, "looks": looks, "alpha": alpha,
        "reps_null": reps_null, "reps_alt": reps_alt,
        "null_naive_fpr": n_naive, "null_obf_fpr": n_obf,
        "alt_naive_rate": a_naive, "alt_obf_power": a_obf,
        "alt_obf_mean_looks_used": a_looks,
    }
