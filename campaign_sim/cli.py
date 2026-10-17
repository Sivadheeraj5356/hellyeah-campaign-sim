"""Command-line interface.

  python -m campaign_sim simulate --days 28 --out sim.csv
  python -m campaign_sim bounds --looks 5
  python -m campaign_sim validate          # binomial-model MC, standard config
  python -m campaign_sim validate-sim      # full-simulator pipeline MC, standard config
  python -m campaign_sim report --out report.html
"""
from __future__ import annotations

import argparse
import json

from .simulator import CampaignSimulator, CreativeSpec, to_csv
from .experiment import obf_bounds, sample_size_per_arm
from .validation import run_validation, validate_via_simulator
from .report import render_report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="campaign_sim")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("simulate", help="simulate a two-creative campaign to CSV")
    p.add_argument("--days", type=int, default=28)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="simulation.csv")

    p = sub.add_parser("bounds", help="print OBF boundaries for K equally-spaced looks")
    p.add_argument("--looks", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.05)

    p = sub.add_parser("validate", help="Monte Carlo validation of FPR and power")
    p.add_argument("--reps-null", type=int, default=20000)
    p.add_argument("--reps-alt", type=int, default=10000)
    p.add_argument("--looks", type=int, default=20)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("validate-sim", help="pipeline validation on full simulator output")
    p.add_argument("--reps-null", type=int, default=2000)
    p.add_argument("--reps-alt", type=int, default=1000)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("report", help="render the single-file HTML demo report")
    p.add_argument("--out", default="report.html")
    p.add_argument("--seed", type=int, default=7)

    args = ap.parse_args(argv)

    if args.cmd == "simulate":
        sim = CampaignSimulator([
            CreativeSpec("Control (A)", 0.020, 0.100),
            CreativeSpec("Challenger (B)", 0.022, 0.115),
        ], seed=args.seed)
        rows = sim.run(args.days)
        with open(args.out, "w") as fh:
            fh.write(to_csv(rows))
        print(f"wrote {len(rows)} rows to {args.out}")
    elif args.cmd == "bounds":
        fracs = [(i + 1) / args.looks for i in range(args.looks)]
        bs = obf_bounds(fracs, alpha=args.alpha)
        for i, (f, b) in enumerate(zip(fracs, bs), 1):
            btxt = "inf" if b == float("inf") else f"{b:.4f}"
            print(f"look {i:2d}  info {f:6.1%}  boundary {btxt}")
    elif args.cmd == "validate":
        null, alt = run_validation(reps_null=args.reps_null,
                                   reps_alt=args.reps_alt, looks=args.looks)
        out = {"null": null.to_dict(), "alt": alt.to_dict(),
               "design_n_per_arm_total": sample_size_per_arm(0.10, 0.015)}
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"naive peeking FPR : {null.naive_peeking_fpr:.3%}  (nominal {null.alpha:.0%})")
            print(f"OBF FPR           : {null.obf_fpr:.3%}  (nominal {null.alpha:.0%})")
            print(f"OBF power (+1.5pt): {alt.obf_power:.3%}  (fixed-horizon design target 80%)")
            print(f"avg looks used    : {alt.obf_mean_looks_used:.1f} of {alt.looks}")
    elif args.cmd == "validate-sim":
        res = validate_via_simulator(reps_null=args.reps_null, reps_alt=args.reps_alt)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"model             : {res['model']}")
            print(f"naive peeking FPR : {res['null_naive_fpr']:.3%}  (nominal {res['alpha']:.0%}, {res['looks']} weekly looks)")
            print(f"OBF FPR           : {res['null_obf_fpr']:.3%}  (nominal {res['alpha']:.0%})")
            print(f"OBF power (+1.5pt): {res['alt_obf_power']:.3%}")
            print(f"avg looks used    : {res['alt_obf_mean_looks_used']:.1f} of {res['looks']}")
    elif args.cmd == "report":
        summary = render_report(args.out, seed=args.seed)
        print(f"wrote {args.out}")
        print(f"binomial MC : naive FPR {summary['null']['naive_peeking_fpr']:.3%}, "
              f"OBF FPR {summary['null']['obf_fpr']:.3%}, "
              f"OBF power {summary['alt']['obf_power']:.3%}")
        print(f"simulator MC: naive FPR {summary['sim']['null_naive_fpr']:.3%}, "
              f"OBF FPR {summary['sim']['null_obf_fpr']:.3%}, "
              f"OBF power {summary['sim']['alt_obf_power']:.3%}")


if __name__ == "__main__":
    main()
