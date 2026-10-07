"""Experiment B: Training job agents and a central planner on one machine.

All three schemes use the same two-feature scoring rule (log w, log p) and plain policy-gradient
learning (REINFORCE with Adam, 150 updates, 16 random instances per update).
  central - a central planner samples a complete sequence (Plackett-Luce)
  team    - job agents swap with neighbours in alternating rounds and share one team reward
  priced  - as team, but each swap decision is rewarded with its own contribution (compensation price)
Every 10 updates, the greedy policy is evaluated on 30 fixed test instances. The measure is the gap to the
optimal schedule, which on one machine is given by Smith's rule (largest w/p first).
Five independent training runs per scheme and queue length (20, 100, 500 jobs).
Output: results/training_runs.json
"""
import json
import time
import numpy as np


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


class Adam:
    def __init__(self, d, lr):
        self.m = np.zeros(d); self.v = np.zeros(d); self.t = 0; self.lr = lr

    def step(self, theta, g):
        self.t += 1; b1, b2 = 0.9, 0.999
        self.m = b1 * self.m + (1 - b1) * g; self.v = b2 * self.v + (1 - b2) * g * g
        mh = self.m / (1 - b1 ** self.t); vh = self.v / (1 - b2 ** self.t)
        return theta + self.lr * mh / (np.sqrt(vh) + 1e-8)


def batch_instances(B, n, rng):
    p = rng.uniform(1, 10, (B, n)); w = rng.uniform(1, 10, (B, n))
    return p, w


def seq_cost(seq, p, w):
    P = np.take_along_axis(p, seq, 1); W = np.take_along_axis(w, seq, 1)
    return (W * np.cumsum(P, 1)).sum(1)


def optimal_cost(p, w):
    seq = np.argsort(-(np.log(w) - np.log(p)), axis=1)   # Smith's rule
    return seq_cost(seq, p, w)


def trading_episode(theta, p, w, rng, K, greedy=False):
    B, n = p.shape
    seq = np.argsort(rng.random((B, n)), 1)
    C0 = seq_cost(seq, p, w)
    lw, lp = np.log(w), np.log(p)
    S_team = np.zeros((B, 2)); G_pr = []
    for t in range(K):
        f = np.arange(t % 2, n - 1, 2)
        jf, jr = seq[:, f], seq[:, f + 1]
        x = np.stack([np.take_along_axis(lw, jr, 1) - np.take_along_axis(lw, jf, 1),
                      np.take_along_axis(lp, jr, 1) - np.take_along_axis(lp, jf, 1)], -1)
        pi = sig(x @ theta)
        a = (pi > 0.5) if greedy else (rng.random(pi.shape) < pi)
        if not greedy:
            sx = (a - pi)[..., None] * x
            S_team += sx.sum(1)
            delta = (np.take_along_axis(w, jr, 1) * np.take_along_axis(p, jf, 1)
                     - np.take_along_axis(w, jf, 1) * np.take_along_axis(p, jr, 1)) / C0[:, None]
            G_pr.append((sx, a * delta))
        newf = np.where(a, jr, jf); newr = np.where(a, jf, jr)
        seq[:, f] = newf; seq[:, f + 1] = newr
    C1 = seq_cost(seq, p, w)
    return C0, C1, S_team, G_pr


def central_episode(theta, p, w, rng, greedy=False):
    B, n = p.shape
    phi = np.stack([np.log(w), np.log(p)], -1)
    s = phi @ theta
    ref = np.argsort(rng.random((B, n)), 1); C0 = seq_cost(ref, p, w)
    if greedy:
        seq = np.argsort(-s, 1)
        return C0, seq_cost(seq, p, w), None
    gum = -np.log(-np.log(rng.random((B, n))))
    seq = np.argsort(-(s + gum), 1)
    ss = np.take_along_axis(s, seq, 1); ph = np.take_along_axis(phi, seq[..., None], 1)
    e = np.exp(ss - ss.max(1, keepdims=True))
    rc_e = np.cumsum(e[:, ::-1], 1)[:, ::-1]; rc_ep = np.cumsum((e[..., None] * ph)[:, ::-1], 1)[:, ::-1]
    score = (ph - rc_ep / rc_e[..., None]).sum(1)
    return C0, seq_cost(seq, p, w), score


def train(method, n, updates=150, B=16, lr=0.05, seed=0, eval_every=10):
    rng = np.random.default_rng(seed)
    theta = np.zeros(2); opt = Adam(2, lr); b = 0.0; nb = 0
    ptest, wtest = batch_instances(30, n, np.random.default_rng(999)); Copt = optimal_cost(ptest, wtest)
    curve = []
    K = n
    for u in range(updates + 1):
        if u % eval_every == 0:
            if method == "central":
                _, C, _ = central_episode(theta, ptest, wtest, np.random.default_rng(5), greedy=True)
            else:
                _, C, _, _ = trading_episode(theta, ptest, wtest, np.random.default_rng(5), K, greedy=True)
            curve.append((u * B, float(np.mean((C - Copt) / Copt))))
        if u == updates:
            break
        p, w = batch_instances(B, n, rng)
        if method == "central":
            C0, C1, score = central_episode(theta, p, w, rng)
            R = (C0 - C1) / C0
            g = ((R - b)[:, None] * score).mean(0)
            b = 0.9 * b + 0.1 * R.mean() if nb else R.mean(); nb = 1
        elif method == "team":
            C0, C1, S, _ = trading_episode(theta, p, w, rng, K)
            R = (C0 - C1) / C0
            g = ((R - b)[:, None] * S).mean(0)
            b = 0.9 * b + 0.1 * R.mean() if nb else R.mean(); nb = 1
        else:
            C0, C1, S, G_pr = trading_episode(theta, p, w, rng, K)
            rs = np.concatenate([r.ravel() for _, r in G_pr]); rb = rs.mean()
            g = sum(((r - rb)[..., None] * sx).sum(1) for sx, r in G_pr).mean(0)
        theta = opt.step(theta, g)
    return theta, curve


def main():
    t0 = time.time()
    out = {}
    for n in [20, 100, 500]:
        for method in ["central", "team", "priced"]:
            runs = []
            for seed in range(5):
                th, cv = train(method, n, seed=100 + seed)
                runs.append({"theta": th.tolist(), "curve": cv})
            out[f"{method}_{n}"] = runs
            gaps = [r["curve"][-1][1] * 100 for r in runs]
            print(f"{method:8s} n = {n:3d}: final gap to optimum (%) = {np.round(gaps, 3)} ({time.time() - t0:.0f} s)")
    with open("results/training_runs.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
