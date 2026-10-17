"""Synthetic paid-media campaign world.

The simulator generates daily per-creative metrics (impressions, clicks,
conversions, spend) from explicit ground-truth parameters, so experiments run
against it have a known correct answer. Effects modeled:

- budget saturation: impressions are a diminishing-returns curve of daily spend
- creative fatigue: CTR decays with cumulative exposure
- day-of-week seasonality
- binomial sampling noise on clicks and conversions

Nothing here touches a real ad platform. All data is synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# Multiplicative seasonality, Monday..Sunday.
DOW_FACTOR = np.array([0.92, 0.96, 1.00, 1.02, 1.07, 1.14, 1.06])


@dataclass
class CreativeSpec:
    """Ground truth for one ad creative."""
    name: str
    true_ctr: float          # baseline click-through rate before fatigue
    true_cvr: float          # conversion rate given a click (constant)
    fatigue_rate: float = 0.03   # CTR multiplier per day is exp(-fatigue_rate * day)
    daily_budget: float = 500.0  # dollars per day
    cpm: float = 12.0            # cost per 1000 impressions before saturation

    def __post_init__(self):
        if not (0 < self.true_ctr < 1):
            raise ValueError("true_ctr must be in (0, 1)")
        if not (0 < self.true_cvr < 1):
            raise ValueError("true_cvr must be in (0, 1)")
        if self.fatigue_rate < 0:
            raise ValueError("fatigue_rate must be >= 0")
        if self.daily_budget <= 0:
            raise ValueError("daily_budget must be > 0")


@dataclass
class DayRow:
    day: int
    creative: str
    impressions: int
    clicks: int
    conversions: int
    spend: float

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions else 0.0

    @property
    def cvr(self) -> float:
        return self.conversions / self.clicks if self.clicks else 0.0

    @property
    def cpa(self) -> float:
        return self.spend / self.conversions if self.conversions else float("inf")


class CampaignSimulator:
    """Simulates daily metrics for a set of creatives.

    Impressions follow a saturation curve: impressions = cap * (1 - exp(-spend/scale)),
    so doubling budget never doubles reach. The cap is derived from cpm and an
    audience-size proxy so the curve is interpretable.
    """

    def __init__(self, creatives, seed: int = 0, audience_cap: int = 200_000):
        if not creatives:
            raise ValueError("at least one creative is required")
        names = [c.name for c in creatives]
        if len(set(names)) != len(names):
            raise ValueError("creative names must be unique")
        self.creatives = list(creatives)
        self.rng = np.random.default_rng(seed)
        self.audience_cap = audience_cap

    def _daily_impressions(self, c: CreativeSpec, day: int) -> int:
        # Nominal impressions the budget would buy at flat CPM...
        nominal = c.daily_budget / c.cpm * 1000.0
        # ...damped toward the audience cap (saturation), then seasonality.
        sat = self.audience_cap * (1.0 - np.exp(-nominal / self.audience_cap))
        mean = sat * DOW_FACTOR[day % 7]
        return int(self.rng.poisson(mean))

    def run(self, days: int) -> list[DayRow]:
        if days <= 0:
            raise ValueError("days must be > 0")
        rows: list[DayRow] = []
        for d in range(days):
            for c in self.creatives:
                imp = self._daily_impressions(c, d)
                ctr_t = c.true_ctr * np.exp(-c.fatigue_rate * d)
                clicks = int(self.rng.binomial(imp, min(ctr_t, 1.0)))
                conv = int(self.rng.binomial(clicks, c.true_cvr)) if clicks else 0
                rows.append(DayRow(day=d, creative=c.name, impressions=imp,
                                   clicks=clicks, conversions=conv,
                                   spend=c.daily_budget))
        return rows


def to_csv(rows: list[DayRow]) -> str:
    lines = ["day,creative,impressions,clicks,conversions,spend"]
    for r in rows:
        lines.append(f"{r.day},{r.creative},{r.impressions},{r.clicks},{r.conversions},{r.spend:.2f}")
    return "\n".join(lines) + "\n"
