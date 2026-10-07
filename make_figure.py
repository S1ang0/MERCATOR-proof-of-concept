"""Figure of the proof of concept (Figure 2 of the MERCATOR proposal, Part I).
Reads results/signal_to_noise.json and results/training_runs.json, writes figures/proof_of_concept.(png|pdf)."""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlelocation": "left", "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
})
GREY, ORANGE, BLUE, META = "#8C8C8C", "#D08C3A", "#1F5FAD", "#888888"

snr = json.load(open("results/signal_to_noise.json"))
runs = json.load(open("results/training_runs.json"))
ns = np.array(snr["ns"])
schemes = [("central_shared", "central planner", GREY), ("team_shared", "team reward", ORANGE),
           ("priced_shared", "compensation prices", BLUE)]

fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.45})
ax = axes[0]
for key, label, col in schemes:
    med = np.array(snr["median_snr"][key]); iqr = np.array(snr["iqr"][key])
    ax.fill_between(ns, iqr[:, 0], iqr[:, 1], color=col, alpha=0.18, lw=0)
    ax.plot(ns, med, "-o", color=col, ms=3.2, lw=1.6 if key == "priced_shared" else 1.1)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.minorticks_off()
ax.set_xticks([8, 32, 128, 512]); ax.set_xticklabels(["8", "32", "128", "512"])
ax.set_yticks([0.1, 1, 10, 100]); ax.set_yticklabels(["0.1", "1", "10", "100"])
ax.set_ylim(0.075, 900); ax.set_xlim(6.5, 1024 * 6.5)
ax.set_xlabel("Number of job agents"); ax.set_ylabel("Gradient signal-to-noise ratio")
ax.set_title("Prices make learning signals grow with scale")
end = {k: snr["median_snr"][k][-1] for k, _, _ in schemes}
ax.text(1024 * 1.15, end["priced_shared"], "compensation\nprices", color=BLUE, va="center", fontsize=8)
ax.text(1024 * 1.15, end["team_shared"] * 1.45, "team reward", color=ORANGE, va="center", fontsize=8)
ax.text(1024 * 1.15, end["central_shared"] * 0.70, "central planner", color=GREY, va="center", fontsize=8)
ax.text(0.02, 0.98, "\u2191 higher = better", transform=ax.transAxes, fontsize=8, color=META, ha="left", va="top")
ax.text(-0.18, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")

ax = axes[1]
offsets = {"central": -0.16, "team": 0.0, "priced": 0.16}
cols = {"central": GREY, "team": ORANGE, "priced": BLUE}
names = {"central": "central planner", "team": "team reward", "priced": "compensation prices"}
handles = []
for m in ["central", "team", "priced"]:
    for i, n in enumerate([20, 100, 500]):
        vals = np.array([r["curve"][-1][1] for r in runs[f"{m}_{n}"]]) * 100
        xs = i + offsets[m] + np.linspace(-0.035, 0.035, len(vals))
        ax.scatter(xs, vals, s=10, color=cols[m], alpha=0.75, lw=0, zorder=3)
        ax.plot([i + offsets[m] - 0.07, i + offsets[m] + 0.07], [np.median(vals)] * 2, color=cols[m], lw=1.8, zorder=4)
    handles.append(mpl.lines.Line2D([], [], color=cols[m], marker="o", ms=3.8, lw=0, label=names[m]))
ax.set_xticks(range(3)); ax.set_xticklabels(["20", "100", "500"])
ax.set_xlabel("Number of jobs in the queue"); ax.set_ylabel("Gap to optimal schedule (%)")
ax.set_title("Priced agents reach the optimum in all runs")
ax.set_xlim(-0.45, 2.45); ax.set_ylim(-0.08, 1.95)
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, 0.92), handletextpad=0.3, labelspacing=0.6)
ax.text(0.02, 0.98, "\u2193 lower = better", transform=ax.transAxes, fontsize=8, color=META, ha="left", va="top")
ax.text(-0.18, 1.02, "b", transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
meta = {"Author": "Sebastian Lang", "Title": "MERCATOR proof of concept"}
fig.savefig("figures/proof_of_concept.png", metadata=meta)
fig.savefig("figures/proof_of_concept.pdf", metadata=meta)
print("figure written")
