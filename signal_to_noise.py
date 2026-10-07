"""Experiment A: Quality of the learning signal as the number of job agents grows.

One machine, n jobs with processing times p and urgency weights w drawn from U[1, 10].
Realised processing times are noisy: p * LogNormal(0, 0.3).
For three coordination schemes, the script measures the signal-to-noise ratio (SNR) of
single-sample policy-gradient (REINFORCE) estimates at the initial, uniform policy:

  central   - a central planner ranks all jobs (Plackett-Luce ranking policy)
  team      - neighbouring job agents decide on swaps and share one team reward
  priced    - neighbouring job agents decide on swaps and receive a compensation price,
              i.e. their own contribution to the common gain

Each scheme is evaluated with individual parameters per agent ("individual") and with one
scoring rule shared by all agents ("shared", two features: log w and log p).
SNR = ||E[g]||^2 / E||g - E[g]||^2. Values are medians over random instances.
Output: results/signal_to_noise.json
"""
import json
import time
import numpy as np

SIG = 0.3
NS = [8, 16, 32, 64, 128, 256, 512, 1024]


def instance(n, rng):
    p = rng.uniform(1, 10, n)
    w = rng.uniform(1, 10, n)
    return p, w


def snr(G):
    m = G.mean(axis=0)
    var = ((G - m) ** 2).sum(axis=1).mean()
    return float((m ** 2).sum() / var)


def central_individual(p, w, M, rng):
    n = len(p)
    seqs = np.argsort(rng.random((M, n)), axis=1)
    preal = p[None, :] * rng.lognormal(0, SIG, (M, n))
    P = np.take_along_axis(preal, seqs, axis=1)
    W = w[seqs]
    C = (W * np.cumsum(P, axis=1)).sum(axis=1)
    b = C.mean()
    pos = np.empty_like(seqs)
    rows = np.arange(M)[:, None]
    pos[rows, seqs] = np.arange(n)[None, :]
    H = np.concatenate([[0.0], np.cumsum(1.0 / (n - np.arange(n)))])
    score = 1.0 - H[pos + 1]
    return snr((C - b)[:, None] * score)


def central_shared(p, w, M, rng):
    n = len(p)
    phi = np.stack([np.log(w), np.log(p)], axis=1)
    seqs = np.argsort(rng.random((M, n)), axis=1)
    preal = p[None, :] * rng.lognormal(0, SIG, (M, n))
    P = np.take_along_axis(preal, seqs, axis=1)
    W = w[seqs]
    C = (W * np.cumsum(P, axis=1)).sum(axis=1)
    b = C.mean()
    F = phi[seqs]
    rev_cum = np.cumsum(F[:, ::-1, :], axis=1)[:, ::-1, :]
    cnt = (n - np.arange(n))[None, :, None]
    score = (F - rev_cum / cnt).sum(axis=1)
    return snr((C - b)[:, None] * score)


def local_trading(p, w, M, rng, reward="team", shared=False):
    """Neighbouring pairs decide whether to swap. reward = 'team' or 'priced'."""
    n = len(p) - (len(p) % 2)
    order = rng.permutation(len(p))[:n]
    front, rear = order[0::2], order[1::2]
    pf = p[front][None, :] * rng.lognormal(0, SIG, (M, n // 2))
    pr = p[rear][None, :] * rng.lognormal(0, SIG, (M, n // 2))
    delta = w[rear][None, :] * pf - w[front][None, :] * pr   # gain of a swap for all jobs
    a = (rng.random((M, n // 2)) < 0.5).astype(float)
    s = a - 0.5
    if reward == "team":
        r = np.repeat((a * delta).sum(axis=1, keepdims=True), n // 2, axis=1)
    else:
        r = a * delta                                           # own contribution = compensation price
    b = r.mean(axis=0, keepdims=True)
    g = s * (r - b)
    if not shared:
        return snr(g)
    x = np.stack([np.log(w[rear]) - np.log(w[front]), np.log(p[rear]) - np.log(p[front])], axis=1)
    return snr(g @ x)


def main():
    rng = np.random.default_rng(7)
    keys = ["central_individual", "team_individual", "priced_individual", "central_shared", "team_shared", "priced_shared"]
    res = {k: [] for k in keys}
    res_q = {k: [] for k in keys}
    t0 = time.time()
    for n in NS:
        reps = 20 if n <= 256 else 8
        M = 20000 if n <= 256 else 6000
        acc = {k: [] for k in keys}
        for _ in range(reps):
            p, w = instance(n, rng)
            acc["central_individual"].append(central_individual(p, w, M, rng))
            acc["central_shared"].append(central_shared(p, w, M, rng))
            acc["team_individual"].append(local_trading(p, w, M, rng, "team", False))
            acc["priced_individual"].append(local_trading(p, w, M, rng, "priced", False))
            acc["team_shared"].append(local_trading(p, w, M, rng, "team", True))
            acc["priced_shared"].append(local_trading(p, w, M, rng, "priced", True))
        for k in keys:
            res[k].append(float(np.median(acc[k])))
            res_q[k].append([float(np.quantile(acc[k], 0.25)), float(np.quantile(acc[k], 0.75))])
        print(f"n = {n:5d} done ({time.time() - t0:.0f} s)")
    slopes = {k: round(float(np.polyfit(np.log(NS), np.log(res[k]), 1)[0]), 3) for k in keys}
    out = {"ns": NS, "median_snr": res, "iqr": res_q, "loglog_slopes": slopes, "noise_sigma": SIG,
           "setup": "single machine; p, w ~ U[1, 10]; realised p = p * LogNormal(0, 0.3); policies at uniform "
                    "initialisation; SNR = ||E g||^2 / E||g - E g||^2 of single-sample REINFORCE estimates; "
                    "per-agent constant baselines; medians over random instances (20, or 8 for n >= 512)"}
    with open("results/signal_to_noise.json", "w") as f:
        json.dump(out, f, indent=1)
    print("log-log slopes:", slopes)


if __name__ == "__main__":
    main()
