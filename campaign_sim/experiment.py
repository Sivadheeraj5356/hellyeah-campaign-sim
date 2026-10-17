"""Experiment analysis for creative A/B tests.

- fixed-horizon two-proportion z-test and sample-size planning
- group-sequential monitoring with O'Brien-Fleming-style boundaries computed
  from the Lan-DeMets alpha-spending function (Armitage-McPherson-Rowe
  recursion on a grid; no lookup tables)
- the point of the sequential part: checking results daily with an unadjusted
  p < 0.05 rule inflates false positives far beyond 5%; the spending-function
  boundaries keep the overall rate at alpha. The validation module measures
  both empirically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def sample_size_per_arm(p_base: float, mde_abs: float,
                        alpha: float = 0.05, power: float = 0.80) -> int:
    """Per-arm sample size (clicks) for a two-sided two-proportion test.

    p_base: baseline rate. mde_abs: absolute minimum detectable effect.
    """
    if not (0 < p_base < 1) or mde_abs <= 0 or p_base + mde_abs >= 1:
        raise ValueError("need 0 < p_base < p_base + mde_abs < 1")
    p1, p2 = p_base, p_base + mde_abs
    p_bar = (p1 + p2) / 2.0
    za = norm.ppf(1 - alpha / 2)
    zb = norm.ppf(power)
    n = (za * math.sqrt(2 * p_bar * (1 - p_bar))
         + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2
    return math.ceil(n)


def z_test_two_proportions(x_t: int, n_t: int, x_c: int, n_c: int):
    """Two-sided z-test for difference of proportions. Returns (z, p)."""
    if n_t <= 0 or n_c <= 0:
        raise ValueError("sample sizes must be positive")
    p_pool = (x_t + x_c) / (n_t + n_c)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    if se == 0:
        return 0.0, 1.0
    z = (x_t / n_t - x_c / n_c) / se
    return z, 2 * (1 - norm.cdf(abs(z)))


def alpha_spent_obf(t: float, alpha: float) -> float:
    """Lan-DeMets spending function approximating O'Brien-Fleming.

    alpha*(t) = 2 * (1 - Phi(z_{alpha/2} / sqrt(t))), for information fraction t.
    """
    if t <= 0:
        return 0.0
    return float(2 * (1 - norm.cdf(norm.ppf(1 - alpha / 2) / math.sqrt(t))))


def obf_bounds(info_fractions, alpha: float = 0.05,
               grid_h: float = 0.005, grid_w: float = 12.0) -> list[float]:
    """Two-sided group-sequential z-boundaries at the given information fractions.

    Recursive numerical integration: track the density of the partial-sum
    process S(t_i) conditioned on not having crossed earlier boundaries,
    convolve with the independent Normal(0, dt) increment, and solve for the
    boundary whose fresh crossing mass equals the incremental spent alpha.
    """
    ts = [float(t) for t in info_fractions]
    if not ts or any(t <= 0 for t in ts):
        raise ValueError("information fractions must be positive")
    if any(ts[i] >= ts[i + 1] for i in range(len(ts) - 1)):
        raise ValueError("information fractions must be strictly increasing")
    if ts[-1] > 1 + 1e-9:
        raise ValueError("final information fraction cannot exceed 1")

    x = np.arange(-grid_w, grid_w + grid_h, grid_h)
    f = None  # unnormalized density of S(t_i) given no earlier crossing
    t_prev = 0.0
    spent_prev = 0.0
    bounds: list[float] = []

    for t in ts:
        dt = t - t_prev
        sd = math.sqrt(dt)
        if f is None:
            fnew = norm.pdf(x, scale=sd)
        else:
            fnew = np.convolve(f, norm.pdf(x, scale=sd), mode="same") * grid_h
        cdf = np.cumsum(fnew) * grid_h
        total = float(cdf[-1])
        target = alpha_spent_obf(t, alpha) - spent_prev
        if target <= 0:
            # So little information that the spent alpha rounds to zero: no
            # crossing is possible at this look (infinite boundary), and the
            # survival density passes through untruncated.
            bounds.append(float("inf"))
            f = fnew
            t_prev = t
            continue

        lo, hi = 0.0, grid_w
        for _ in range(60):  # bisection on the boundary b
            mid = (lo + hi) / 2
            i_lo = int(np.searchsorted(x, -mid))
            i_hi = int(np.searchsorted(x, mid))
            crossed = total - (cdf[i_hi] - cdf[i_lo])
            if crossed > target:
                lo = mid
            else:
                hi = mid
        b = (lo + hi) / 2
        bounds.append(float(b / math.sqrt(t)))
        f = np.where(np.abs(x) <= b, fnew, 0.0)
        spent_prev += target
        t_prev = t
    return bounds


@dataclass
class LookResult:
    look: int
    info_fraction: float
    z: float
    bound: float
    crossed: bool


def sequential_analysis(looks, alpha: float = 0.05) -> list[LookResult]:
    """Evaluate cumulative (x_t, n_t, x_c, n_c) per look against OBF bounds.

    looks: list of dicts/tuples with cumulative counts and information
    fraction per look: (info_fraction, x_t, n_t, x_c, n_c).
    """
    fracs = [l[0] for l in looks]
    bounds = obf_bounds(fracs, alpha=alpha)
    out = []
    for (frac, x_t, n_t, x_c, n_c), bound in zip(looks, bounds):
        z, _ = z_test_two_proportions(x_t, n_t, x_c, n_c)
        out.append(LookResult(look=len(out) + 1, info_fraction=frac, z=z,
                              bound=bound, crossed=abs(z) > bound))
    return out
