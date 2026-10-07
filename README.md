# MERCATOR - Proof of Concept: Compensation Prices and Learning Signals in a Queue of Job Agents

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23212939.svg)](https://doi.org/10.5281/zenodo.23212939)

This repository contains the proof-of-concept computations for the research project
**MERCATOR – Markets or Plans? Economic Institutions for Self-Organising Production with Learning Agents**.

MERCATOR treats every job in a factory as a learning agent that trades places in the machine queues.
The coordination rules of this economy (who gets which budget, how places are traded, who pays whom)
decide how well the jobs learn. The proof of concept tests one core assumption in the simplest case,
one machine with one queue:

> If every job pays a compensation price equal to the delay it causes to the other jobs, each job keeps
> a clear learning signal as the number of jobs grows. Under a shared team reward or a central planner,
> the learning signal becomes noisier with every additional job.

## What is computed

**Experiment A – quality of the learning signal** (`signal_to_noise.py`, about 1 minute).
One machine, 8 to 1,024 jobs with processing times and urgency weights drawn from U[1, 10] and noisy
realised processing times. For three coordination schemes, the script measures the signal-to-noise ratio
of single-sample policy-gradient estimates at the start of learning:

- *central planner*: one agent ranks all jobs,
- *team reward*: neighbouring job agents decide on swaps and share one reward,
- *compensation prices*: neighbouring job agents decide on swaps and are rewarded with their own contribution.

Each scheme is evaluated with individual parameters per agent and with one scoring rule shared by all agents.

**Experiment B – training runs** (`training_runs.py`, about 3 to 4 minutes).
All three schemes learn the same two-feature scoring rule with plain policy-gradient learning
(150 updates, 16 random instances per update, five independent runs, 20, 100 and 500 jobs).
The measure is the gap to the optimal schedule, which on one machine is known (Smith's rule).

`make_figure.py` draws both results (Figure 2 of the proposal, Part I).

## Main results

| Scheme | Slope of signal quality vs. number of jobs (log–log), individual parameters | Shared scoring rule |
|---|---|---|
| Central planner | −0.99 | −0.07 |
| Team reward | −1.02 | −0.07 |
| Compensation prices | −0.01 | +0.95 |

In Experiment B, agents with compensation prices came within 0.001 % of the optimal schedule in all 15 runs.
Agents with a team reward stayed 0.01 % to 1.8 % above the optimum, with the largest gaps for 500 jobs.
The central planner came close to the optimum in 3 of 15 runs and stayed up to 0.41 % above it in the others.

## What the proof of concept does not show

It covers one machine, full information and one round of trading. It does not show that the mechanism works
in multi-stage production, with jobs that arrive over time, with private urgency or with many rounds of trading.
These questions are the subject of MERCATOR.

## How to run

```bash
pip install -r requirements.txt
python signal_to_noise.py   # writes results/signal_to_noise.json
python training_runs.py     # writes results/training_runs.json
python make_figure.py       # writes figures/proof_of_concept.png and .pdf
```

All random seeds are fixed. With numpy 2.4.6, the scripts reproduce the files in `results/` exactly.
With other numpy versions, Experiment A gives identical values. In Experiment B, single team-reward runs
can end at a different gap, because small numerical differences change individual random decisions
during training. The runs with compensation prices are not affected.

## Licence and citation

Code and results are released under the MIT licence. Please cite this repository with the metadata in
`CITATION.cff` (DOI: [10.5281/zenodo.23212939](https://doi.org/10.5281/zenodo.23212939)).
