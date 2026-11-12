"""Analysis of the 40 runs (BC, DAgger, SAC, RLPD x 10 seeds)

    python analysis.py            # reads results/*.json, writes figures/ and results/summary.csv

Measures (per run), all in lap steps (lower = faster, 1500 = lap not finished):
- final lap: mean of the evaluations at 45k, 47.5k and 50k online steps
  (BC is never retrained: its single evaluation)
- mean lap over training: mean of the evaluations at 0, 2.5k, ..., 50k steps
  (area under the learning curve: sample efficiency)
- steps to expert level: first evaluation step where the lap takes at most
  5% more steps than the expert (614 steps); never reached = infinity

DAgger is evaluated after each iteration (irregular steps): its value at step x
is the evaluation of the last policy trained with at most x online steps.
"""

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

EXPERT_LAP = 585                       # clean team4 lap through our wrapper (noise 0)
NOT_FINISHED = 1500                    # episode limit: lap not finished
GRID = np.arange(0, 50_001, 2_500)     # common evaluation steps
FINAL = GRID[-3:]                      # 45k, 47.5k, 50k
LEVEL = 1.05 * EXPERT_LAP              # "expert level": within 5% of the expert
METHODS = ["bc", "dagger", "sac", "rlpd"]
LABELS = {"bc": "BC", "dagger": "DAgger", "sac": "SAC", "rlpd": "RLPD"}
COLORS = {"bc": "#7f7f7f", "dagger": "#1f77b4", "sac": "#ff7f0e", "rlpd": "#2ca02c"}
PAIRS = [("dagger", "rlpd"), ("bc", "dagger"), ("sac", "rlpd")]


def lap(e):
    return e["lap_steps"] if e["finished"] else NOT_FINISHED


def on_grid(evaluations):
    """Lap steps at each step of GRID (last evaluation at or before that step)"""
    steps = np.array([e["step"] for e in evaluations])
    laps = np.array([lap(e) for e in evaluations])
    _, keep = np.unique(steps[::-1], return_index=True)   # duplicated steps: keep the last one
    keep = len(steps) - 1 - keep
    steps, laps = steps[keep], laps[keep]
    idx = np.searchsorted(steps, GRID, side="right") - 1
    return laps[idx]


def load():
    curves, rows = {}, []
    for m in METHODS:
        curves[m] = []
        for f in sorted(glob.glob(f"results/{m}-s*.json")):
            ev = json.load(open(f))["evaluations"]
            if m == "bc":
                c = np.full(len(GRID), lap(ev[0]), dtype=float)
            else:
                c = on_grid(ev).astype(float)
            curves[m].append(c)
            reached = np.nonzero(c <= LEVEL)[0]
            rows.append({
                "method": m, "seed": int(Path(f).stem.split("-s")[1]),
                "final_lap": c[-3:].mean() if m != "bc" else c[0],
                "mean_lap_training": c.mean(),
                "steps_to_expert_level": GRID[reached[0]] if len(reached) else np.inf,
                "finished_final": bool(c[-1] < NOT_FINISHED),
            })
        curves[m] = np.array(curves[m])
    return curves, pd.DataFrame(rows)


def tests(df):
    """Two-sided Mann-Whitney U tests, Bonferroni over all the comparisons"""
    measures = ["final_lap", "mean_lap_training"]
    out = []
    for a, b in PAIRS:
        for meas in measures:
            x = df[df.method == a][meas].to_numpy()
            y = df[df.method == b][meas].to_numpy()
            out.append({"comparison": f"{LABELS[a]} vs {LABELS[b]}", "measure": meas,
                        f"median_a": np.median(x), "median_b": np.median(y),
                        "p_mannwhitney": stats.mannwhitneyu(x, y, alternative="two-sided").pvalue,
                        "p_welch": stats.ttest_ind(x, y, equal_var=False).pvalue})
    out = pd.DataFrame(out)
    out["alpha_bonferroni"] = 0.05 / len(out)
    out["significant"] = out.p_mannwhitney < out.alpha_bonferroni
    return out


def plot(curves, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    fig.suptitle("Lap time and success along training on abyss (10 seeds per method)")
    x = GRID / 1000
    for m in METHODS:
        c = curves[m]
        med, q1, q3 = np.median(c, 0), np.percentile(c, 25, 0), np.percentile(c, 75, 0)
        ax1.plot(x, med, color=COLORS[m], label=LABELS[m])
        ax1.fill_between(x, q1, q3, color=COLORS[m], alpha=0.2, linewidth=0)
        ax2.plot(x, (c < NOT_FINISHED).mean(0) * 100, color=COLORS[m], label=LABELS[m])
    ax1.axhline(EXPERT_LAP, color="black", linestyle="--", linewidth=1, label="Expert (team 4)")
    ax1.set_ylim(550, 700)
    ax1.set_xlabel("Online environment steps (thousands)")
    ax1.set_ylabel("Lap time (steps, lower is faster)")
    ax1.set_title("(a) Lap time: median and interquartile range", fontsize=10)
    ax1.legend(fontsize=8)
    ax2.set_ylim(-5, 105)
    ax2.set_xlabel("Online environment steps (thousands)")
    ax2.set_ylabel("Runs finishing the lap (%)")
    ax2.set_title("(b) Success rate", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    print(f"Figure saved in {path}")


if __name__ == "__main__":
    curves, df = load()
    Path("figures").mkdir(exist_ok=True)
    df.to_csv("results/summary.csv", index=False)

    pd.set_option("display.width", 140)
    print("Runs per method:", df.groupby("method").size().to_dict(), "\n")
    summary = df.groupby("method").agg(
        final_lap_mean=("final_lap", "mean"), final_lap_std=("final_lap", "std"),
        final_lap_median=("final_lap", "median"),
        mean_lap_training=("mean_lap_training", "median"),
        steps_to_expert_level=("steps_to_expert_level", "median"),
        faster_than_expert=("final_lap", lambda v: int((v < EXPERT_LAP).sum())),
        finished=("finished_final", "sum")).loc[METHODS]
    print(summary.round(1), "\n")
    print(tests(df).round(5).to_string(index=False))
    plot(curves, "figures/curves.pdf")
