"""Analysis of the 40 runs (BC, DAgger, SAC, RLPD x 10 seeds)

    python analysis.py      # reads results/*.json
                            # writes figures/performance.pdf, figures/learning_speed.pdf,
                            # results/summary.csv, results/tests.csv and results/table.tex

Lap times are given in seconds of game time (one environment step = 0.1 s;
lower = faster). A lap that is not finished counts as 150 s (the 1500-step
episode limit). Expert: 58.5 s. "Training steps" are the online environment
steps used by a method to learn.

Measures per run:
- final lap time: mean of the evaluations at 45k, 47.5k and 50k training steps
  (BC is never retrained: its single evaluation)
- learning speed: first evaluation (every 2.5k training steps) at which the lap
  is at most 5% slower than the expert (<= 61.4 s), and at most 2.5% slower
  (<= 59.9 s). Reaching the expert's own time (58.5 s) is reported as a
  sensitivity check only.

DAgger is evaluated after each iteration (irregular steps): its value at step x
is the evaluation of the last policy trained with at most x training steps.

Tests: two-sided Mann-Whitney U on the per-run values, Bonferroni correction
over the 7 comparisons (BC vs DAgger is only compared on the final lap time:
BC does not learn online, so it has no learning speed).
"""

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

STEP_S = 0.1                           # game time of one environment step (s)
EXPERT_LAP = 585                       # clean team4 lap through our wrapper (steps)
NOT_FINISHED = 1500                    # episode limit: lap not finished
GRID = np.arange(0, 50_001, 2_500)     # common evaluation steps
N_FINAL = 3                            # final lap = mean of the last 3 evaluations
THRESHOLDS = {"5%": 1.05, "2.5%": 1.025, "expert": 1.0}
NEVER = np.inf

METHODS = ["bc", "dagger", "sac", "rlpd"]
LABELS = {"bc": "BC", "dagger": "DAgger", "sac": "SAC", "rlpd": "RLPD"}
# First four slots of a colour-blind-checked categorical palette
COLORS = {"bc": "#2a78d6", "dagger": "#eb6834", "sac": "#1baf7a", "rlpd": "#eda100"}
TESTS = [("dagger", "rlpd", "final_lap"), ("bc", "dagger", "final_lap"),
         ("sac", "rlpd", "final_lap"),
         ("dagger", "rlpd", "steps_to_5%"), ("sac", "rlpd", "steps_to_5%"),
         ("dagger", "rlpd", "steps_to_2.5%"), ("sac", "rlpd", "steps_to_2.5%")]
ALPHA = 0.05 / len(TESTS)

# Computer Modern (the default LaTeX font) for all the text of the figures
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"],
                     "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
                     "axes.unicode_minus": False, "pdf.fonttype": 42})


def lap(e):
    return e["lap_steps"] if e["finished"] else NOT_FINISHED


def threshold_s(t):
    """Largest lap time (s) within the threshold: laps are whole numbers of steps"""
    return np.floor(t * EXPERT_LAP) * STEP_S


def on_grid(evaluations):
    """Lap steps at each step of GRID (last evaluation at or before that step)"""
    steps = np.array([e["step"] for e in evaluations])
    laps = np.array([lap(e) for e in evaluations])
    # rlpd.py may evaluate twice at the same step: keep the last evaluation
    keep = np.array([i == len(steps) - 1 or steps[i + 1] != steps[i] for i in range(len(steps))])
    steps, laps = steps[keep], laps[keep]
    return laps[np.searchsorted(steps, GRID, side="right") - 1]


def first_step_below(curve, threshold):
    reached = np.nonzero(curve <= threshold * EXPERT_LAP)[0]
    return GRID[reached[0]] if len(reached) else NEVER


def load():
    """Learning curves (lap time in s on GRID) and per-run measures"""
    curves, rows = {m: [] for m in METHODS}, []
    for m in METHODS:
        for f in sorted(glob.glob(f"results/{m}-s*.json")):
            ev = json.load(open(f))["evaluations"]
            c = np.full(len(GRID), lap(ev[0])) if m == "bc" else on_grid(ev)
            curves[m].append(c * STEP_S)
            row = {"method": m, "seed": int(Path(f).stem.split("-s")[1]),
                   "final_lap": (c[0] if m == "bc" else c[-N_FINAL:].mean()) * STEP_S,
                   "best_lap": min(lap(e) for e in ev) * STEP_S}
            for name, t in THRESHOLDS.items():
                row[f"steps_to_{name}"] = NEVER if m == "bc" else first_step_below(c, t)
            # evaluations where the lap is not finished, after the first finished one
            first = np.argmax(c < NOT_FINISHED)
            row["failures_after_first_finish"] = int((c[first:] == NOT_FINISHED).sum())
            rows.append(row)
        curves[m] = np.array(curves[m], dtype=float)
    return curves, pd.DataFrame(rows)


def values(df, method, measure):
    v = df[df.method == method][measure].to_numpy(dtype=float)
    return np.where(np.isinf(v), 1e9, v)     # "never" ranks after every finite value


def mann_whitney(df, comparisons):
    out = []
    for a, b, meas in comparisons:
        x, y = values(df, a, meas), values(df, b, meas)
        out.append({"comparison": f"{LABELS[a]} vs {LABELS[b]}", "measure": meas,
                    "median_a": np.median(x), "median_b": np.median(y),
                    "p": stats.mannwhitneyu(x, y, alternative="two-sided").pvalue})
    return pd.DataFrame(out)

# Figures

def fmt_p(p):
    return f"$p = {p:.3f}$" if p >= 0.001 else f"$p < 0.001$"


def bracket(ax, x1, x2, y, p, h):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="0.3", linewidth=0.8)
    ax.text((x1 + x2) / 2, y + h, fmt_p(p), ha="center", va="bottom", fontsize=8, color="0.2")


def strip(ax, df, methods, measure, scale=1.0):
    """One point per run (slight horizontal jitter) and a line at the median"""
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        v = df[df.method == m][measure].to_numpy(dtype=float) / scale
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=22, color=COLORS[m],
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="0.15", linewidth=1.5, zorder=4)
    ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods])
    ax.set_xlim(-0.6, len(methods) - 0.4)


def style(axes):
    for ax in axes:
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)


def reference_line(ax, y, text, x, ha):
    ax.axhline(y, color="0.2", linewidth=0.8, zorder=1)
    ax.text(x, y - 0.15, text, ha=ha, va="top", fontsize=7.5, color="0.2")


def plot_performance(curves, df, p, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.suptitle("Lap time of BC, DAgger, SAC and RLPD on abyss (10 seeds each)", fontsize=11)

    # (a) learning curves: median and interquartile range over the 10 seeds
    ax = axes[0]
    x = GRID / 1000
    for m in METHODS:
        c = curves[m]
        ax.fill_between(x, np.percentile(c, 25, 0), np.percentile(c, 75, 0),
                        color=COLORS[m], alpha=0.18, linewidth=0)
        ax.plot(x, np.median(c, 0), color=COLORS[m], linewidth=1.8, label=LABELS[m])
    reference_line(ax, EXPERT_LAP * STEP_S, "expert (58.5 s)", 0.5, "left")
    ax.set_ylim(56, 68)
    ax.set_xlim(0, 50)
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel("Lap time (s)")
    ax.set_title("(a) Lap time along training", fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="upper right", frameon=False)

    # (b) final lap time, one point per run
    ax = axes[1]
    strip(ax, df, METHODS, "final_lap")
    reference_line(ax, EXPERT_LAP * STEP_S, "expert", -0.55, "left")
    ax.set_ylim(55.5, 70)
    bracket(ax, 0, 1, 66.2, p[("BC vs DAgger", "final_lap")], 0.3)
    bracket(ax, 2, 3, 62.5, p[("SAC vs RLPD", "final_lap")], 0.3)
    bracket(ax, 1, 3, 67.8, p[("DAgger vs RLPD", "final_lap")], 0.3)
    ax.set_ylabel("Final lap time (s)")
    ax.set_title("(b) Final lap time", fontsize=10)

    style(axes)
    fig.tight_layout()
    fig.savefig(path)
    print(f"Figure saved in {path}")


def plot_learning_speed(df, p, path):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
    fig.suptitle("Training steps needed to approach the expert's lap time (10 seeds each)",
                 fontsize=11)
    methods = ["dagger", "sac", "rlpd"]
    for ax, (name, letter) in zip(axes, [("5%", "a"), ("2.5%", "b")]):
        meas = f"steps_to_{name}"
        strip(ax, df, methods, meas, scale=1000)
        bracket(ax, 1, 2, 36, p[("SAC vs RLPD", meas)], 1.2)
        bracket(ax, 0, 2, 41, p[("DAgger vs RLPD", meas)], 1.2)
        ax.set_title(f"({letter}) Within {name} of the expert\n"
                     f"(lap $\\leq$ {threshold_s(THRESHOLDS[name]):.1f} s)", fontsize=10)
    axes[0].set_ylim(-2, 46)
    axes[0].set_ylabel("Training steps to reach the threshold\n(thousands)")
    style(axes)
    fig.tight_layout()
    fig.savefig(path)
    print(f"Figure saved in {path}")

# LaTeX table

def latex_table(df, path):
    def med_range(v):
        v = v[np.isfinite(v)] / 1000
        return f"{np.median(v):.1f} [{v.min():.1f}--{v.max():.1f}]" if len(v) else "--"

    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Method & Final lap (s) & Faster than & Best lap (s) & \multicolumn{2}{c}{Training steps (k) to reach} & Failed laps \\",
             r" & mean $\pm$ std & expert & median & $\leq$ 61.4 s (5\%) & $\leq$ 59.9 s (2.5\%) & after 1st finish \\",
             r"\midrule"]
    for m in METHODS:
        d = df[df.method == m]
        steps = ["--", "--"] if m == "bc" else [
            med_range(d[f"steps_to_{n}"].to_numpy(dtype=float)) for n in ("5%", "2.5%")]
        lines.append(f"{LABELS[m]} & ${d.final_lap.mean():.1f} \\pm {d.final_lap.std():.1f}$ & "
                     f"{int((d.final_lap < EXPERT_LAP * STEP_S).sum())}/{len(d)} & "
                     f"{d.best_lap.median():.1f} & {steps[0]} & {steps[1]} & "
                     f"{int(d.failures_after_first_finish.sum())} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"Table saved in {path}")


if __name__ == "__main__":
    curves, df = load()
    Path("figures").mkdir(exist_ok=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print("Runs per method:", df.groupby("method").size().to_dict(), "\n")
    print(df.groupby("method").agg(
        final_mean_s=("final_lap", "mean"), final_std_s=("final_lap", "std"),
        best_lap_s=("best_lap", "median"),
        faster_than_expert=("final_lap", lambda v: int((v < EXPERT_LAP * STEP_S).sum())),
        steps_to_5=("steps_to_5%", "median"), steps_to_2_5=("steps_to_2.5%", "median"),
        failures=("failures_after_first_finish", "sum")).loc[METHODS].round(2), "\n")

    tests = mann_whitney(df, TESTS)
    tests["significant"] = tests.p < ALPHA
    print(f"Mann-Whitney U, Bonferroni threshold 0.05/{len(TESTS)} = {ALPHA:.4f}")
    print(tests.round(5).to_string(index=False), "\n")
    print("Sensitivity (expert level, not corrected):")
    print(mann_whitney(df, [("sac", "rlpd", "steps_to_expert"),
                            ("dagger", "rlpd", "steps_to_expert")]).round(4).to_string(index=False))

    df.to_csv("results/summary.csv", index=False)
    tests.to_csv("results/tests.csv", index=False)
    latex_table(df, "results/table.tex")
    p = {(r.comparison, r.measure): r.p for r in tests.itertuples()}
    plot_performance(curves, df, p, "figures/performance.pdf")
    plot_learning_speed(df, p, "figures/learning_speed.pdf")
