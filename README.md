# Reading Minds, Imagining, Then Acting: Simulating and Probing Other Agents' Latent World Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23601306.svg)](https://doi.org/10.5281/zenodo.23601306)

Code, frozen splits, aggregate results and trained models for the MSc thesis, by Andrés Daniel Godoy Ortiz. MSc in Artificial Intelligence Applications and Innovation, Imperial College London, 2026.

The thesis is in [`paper/ATOM_thesis_Godoy_2026.pdf`](paper/ATOM_thesis_Godoy_2026.pdf).

## Abstract

Any system that acts on another agent must infer the latent model behind its behaviour, predict how the agent would act in situations not yet observed, and choose how to respond. Our goal is to learn the latent model of a learner to simulate the answers they would give to questions the model has not seen, and, based on that, to choose the questions that reveal the most about them. We bring these problems into one framework and build ATOM, which turns a learner's history into a runnable model of their mind, used as a simulator for training LLMs and RL policies before real interaction. ATOM combines population-level learner modelling with language models: a frozen semantic encoder represents questions by their text, with no item IDs or skill labels; a variational filter maps a history to a Gaussian belief over a learner latent; and a hypernetwork turns each belief sample into a low-rank weight edit of a frozen 4B-parameter LLM. On FoundationalASSIST, ATOM predicts learner success at 0.786 AUC versus 0.512-0.559 for 70-120B LLMs reading histories, within 0.024 AUC of attentive knowledge tracing, which uses skill tags ATOM never sees. It reproduces the learner's exact answer 46.0% of the time; supplying the true outcome class raises this to 57.3%, showing that inferring the learner, not generation, is the bottleneck. On CodeWorkout, it reaches 0.833 AUC against the best published 0.816 on held-out learners, and 0.561 CodeBLEU on TIKTOC against 0.554, zero-shot. The belief is also actionable: belief-aware question-selection policies beat random questioning and belief-blind heuristics; a policy trained entirely inside the simulator transfers in the favourable direction in five of six real-learner settings (underpowered); and an LLM fine-tuned on the simulator's counterfactual value estimates beats random in all six settings, recovering 25-74% of oracle headroom.

## The system

ATOM has two halves: **ATOM-Sim**, the simulator, and **ATOM-Act**, the planner. In the thesis's terms, it is three spaces and two learned maps (Chapter 5).

1. **Space of meanings.** A frozen encoder maps each problem's text to a point, so problems about similar things land near one another however they are phrased. No item identifier, skill tag or prerequisite graph enters the system. Code: `src/atom/codec/`.
2. **Space of minds.** A recurrent variational filter reads a learner's history as a path through the space of meanings and returns a Gaussian belief over a person latent: a location, and a region saying how sure it is. Code: `src/atom/filter/`.
3. **The world model.** A hypernetwork turns each sample of the belief into a low-rank weight edit of a frozen language model (Gemma 4 E4B). The edited model is the simulated learner: given a new problem, it says whether they would succeed and writes what they would write. Several samples give several plausible versions of the same learner, and so an uncertainty. Code: `src/atom/binding/`, `src/atom/heads/`, `src/atom/simulator/`.
4. **ATOM-Act.** The planner treats the belief as its state. It values a question by how much its answer is expected to improve later predictions, trains a policy whose state is the belief, and trains a language model to select questions against the values the simulator computes. Code: `src/atom/planner/`.

The system is instantiated on two corpora: FoundationalASSIST (middle-school mathematics) and CodeWorkout (introductory Java).

## Contributions and where they are in the code

The thesis states eight contributions (Section 1.4). [`scripts/README.md`](scripts/README.md) lists every entry point by chapter.

| | Contribution | Thesis | Code |
|---|---|---|---|
| C1 | A conceptual framework for reading and acting on other minds | Chapter 3 | (no code) |
| C2 | Meaning can do the work a curriculum did | Chapters 5, 6 | `src/atom/codec/`, Chapter 6 scripts |
| C3 | On FoundationalASSIST, a frozen backbone made personal by a weight edit reads a learner better than models many times its size | Chapters 5, 7 | FA-SIM3: `src/atom/simulator/fa_alt2.py`, `infra/fa_sim3_full_run.py` |
| C4 | On CodeWorkout, the same architecture holds against systems built for that corpus, on learners it never trained on | Chapters 5, 7 | the classifier and generative simulator: `scripts/cw_sigma.py`, `infra/cw_sim_full_run.py` |
| C5 | Knowledge tracing and generation are two stages of one computation, and the gap between them is measurable | Chapter 7 | FA-SIM3's evaluation |
| C6 | A simulator of a person is a workable environment for reinforcement learning | Chapters 5, 8 | `scripts/cw_planner_suite.py`, `scripts/rl_2*.py`, `scripts/sim2real_w*.py` |
| C7 | A learned simulator of a person can supply a training signal a language model has no other way to get | Chapter 8 | `scripts/ga_1_tier*.py` |
| C8 | An evaluation discipline: two courts, a sealed lockbox, learner-clustered intervals, pre-registered gates | Chapter 4 | `splits/`, `src/atom/eval/` |

## Headline results

All numbers are from the thesis and are computed on learners the models never trained on. [`RESULTS.md`](RESULTS.md) lists every reported number with its interval, its comparison and the command that produces it.

| What | ATOM | Comparison |
|---|---|---|
| The encoder recovers the curriculum's skill grouping without seeing skill labels (FoundationalASSIST) | 0.819 AUC | 0.749 for raw hidden states |
| The encoder identifies which of 50 problems a submission solves (CodeWorkout) | 0.995 AUC | 0.917 for string similarity |
| Predicting whether a learner gets the next problem right (FoundationalASSIST) | 0.786 AUC | 0.512–0.559 for four frontier language models of 70–120B parameters reading the same history; 0.810 for attentive knowledge tracing, which uses skill tags |
| Reproducing the learner's exact typed answer (FoundationalASSIST) | 46.0%; 57.3% when the true outcome class is given | 44.3% for the best frontier model |
| First-submission correctness (CodeWorkout) | 0.833 AUC | 0.816 best published |
| Writing the learner's program, TIKTOC's own benchmark (CodeWorkout, zero-shot across cohorts) | 0.561 CodeBLEU | 0.554 TIKTOC |
| Choosing diagnostic questions (CodeWorkout, start of course, three questions) | 0.924–0.944 nats for belief-aware policies | 1.072 random; about 1.03 belief-blind heuristics |
| A language model fine-tuned on the simulator's counterfactual values, as a question selector | beats random in 6 of 6 settings, 5 with intervals clear of zero; 25–74% of oracle headroom | the same model untrained loses to random in 36 of 36 |

The negatives are reported as well. On FoundationalASSIST the value-of-information planner does not beat random questioning. A policy trained only inside the simulator transfers in the favourable direction in five of six settings, but no setting is statistically separated from random. The thesis's limitations section (Section 9.2) states the rest.

## Quick start (no data, no GPU, about five minutes)

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/adgodoyo/atom-learner-models.git
cd atom-learner-models
make setup      # creates the pinned environment from uv.lock
make weights    # rebuilds the trained checkpoints (downloads the two large ones from the Hub)
make test       # unit tests; tests that need the corpora or the backbone skip cleanly
make smoke      # end-to-end pipeline on synthetic data
```

## Reproducing a number

1. Obtain the corpora from their owners and place them as described in [`DATA.md`](DATA.md). The repository contains no data.
2. `make data-fa` and `make data-cw` rebuild the manifests and check the frozen splits against their published hashes.
3. Find the number in [`RESULTS.md`](RESULTS.md) and run its command from [`REPRODUCE.md`](REPRODUCE.md). Each row states the hardware it needs: most evaluations run on a laptop CPU; training and text generation need one GPU with 24–48 GB.

> **Warning: the trainers in `infra/` are written for a disposable cloud GPU machine.** To protect against runaway cloud bills, several of them schedule an operating-system shutdown when they start (`sudo shutdown -h +<minutes>`), and some can shut the machine down immediately when they finish. Do not run them on a personal computer or a shared server. The published checkpoints (`make weights`) make retraining unnecessary for every evaluation.

## What is in the repository

| Folder | Contents |
|---|---|
| `src/atom/` | The package: data loaders, the semantic encoder, the belief filter, the low-rank binding and hypernetwork, the heads, the simulators, the planner, baselines and evaluation |
| `scripts/` | One entry point per thesis analysis; [`scripts/README.md`](scripts/README.md) lists them by chapter |
| `infra/` | GPU trainers and generation evaluators (run locally or on any single-GPU machine) |
| `configs/`, `splits/`, `transforms/` | Configurations, the frozen learner and item splits (identifiers only), the frozen encoder transform |
| `artifacts/` | One encoder vector per problem, problem-level labels, and the CodingBat source list |
| `checkpoints/` | The trained models as safetensors with their configs; see [`MODEL_CARD.md`](MODEL_CARD.md) |
| `results/` | Aggregate result files behind the thesis numbers |
| `paper/` | The thesis PDF and the scripts that draw its data figures |
| `tests/` | Unit tests |

The repository holds no learner-level data of any kind, including model outputs such as per-learner beliefs; [`DATA.md`](DATA.md) lists exactly what it does hold. The code uses some development names internally, such as task identifiers in script names and nicknames in the substrate registry. [`MODEL_CARD.md`](MODEL_CARD.md) maps every model to the name the thesis uses.

## Licences

- **Code:** MIT ([`LICENSE`](LICENSE)).
- **Model weights:** research and non-commercial use only. They are derivatives of Gemma 4 E4B and are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and its prohibited-use policy, and they were trained on data licensed CC-BY-NC-4.0. Do not use them to try to identify any learner.
- **Data:** not distributed. See [`DATA.md`](DATA.md) for the official sources and their terms.

## Citation

Please cite the thesis, and this code by its archived version:

- **Thesis:** Godoy Ortiz, A. D. (2026). *Reading Minds, Imagining, Then Acting: Simulating and Probing Other Agents' Latent World Models.* MSc thesis, Imperial College London.
- **Code and models (v1.0.0):** [doi:10.5281/zenodo.23601306](https://doi.org/10.5281/zenodo.23601306). All versions: [doi:10.5281/zenodo.23601305](https://doi.org/10.5281/zenodo.23601305).

[`CITATION.cff`](CITATION.cff) has the same in machine-readable form (GitHub's "Cite this repository" button reads it).

The semantic encoder is adopted from the author's companion paper: Godoy Ortiz (2026), *Meaning Contracts: Local Semantic Geometry in Large Language Models*, [arXiv:2610.30715](https://arxiv.org/abs/2610.30715), with its own code at [github.com/adgodoyo/Meaning_Contracts](https://github.com/adgodoyo/Meaning_Contracts). Its results are not results of this repository.

## Use of generative AI

From the thesis: "I acknowledge the use of Claude (Anthropic, https://www.claude.ai) as a coding assistant during the implementation of this project, including its training and evaluation scripts, and fluency of the written text. The conceptual framework, architecture, experiments, results and analysis in this thesis are my own, and all AI-assisted text was reviewed and revised by me."

Contact: Andrés Godoy, adgodoyo@gmail.com.
