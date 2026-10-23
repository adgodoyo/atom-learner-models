# Model card

These are the models the thesis presents as the configuration that ships (Chapter 5) and the policies behind its planner results (Chapter 8). No earlier or discarded version of any model is published.

## Models

| Release name | What the thesis calls it | Trainable parameters | Where |
|---|---|---:|---|
| `fa-sim3` | The mathematics simulator, FA-SIM3: belief filter (256-dim belief) with its item encoder, hypernetwork, shared low-rank basis (32 atoms × rank 8 over 66 projections) and outcome head; token-loss weight α = 0.3; 14,949 steps (§5.6, Tables 3–4) | 74,737,765 | Hugging Face `adgodoyouk/atom-fa-sim3` |
| `fa-outcome-filter` | The filter and item encoder trained against the outcome objective alone, used as FA-SIM3's warm start (§5.6) and quoted in §6 at 0.6913 | 825,091 | `checkpoints/` |
| `cw-classifier` | The CodeWorkout classifier: attempt encoder, cross-episode filter (32-dim, precision-accumulating width) and outcome head, trained alone for 60 epochs with an eight-sample mixture likelihood; the production predictor for every CodeWorkout classification number and the planner's belief substrate (§5.6, Table 9) | 104,565 | `checkpoints/` |
| `cw-sim` | The CodeWorkout generative simulator: the classifier plus hypernetwork, 16 × 8 basis and test-vector head, trained jointly for two epochs (§5.6, Table 11) | 36,420,836 | Hugging Face `adgodoyouk/atom-cw-sim` |
| `cw-sim-outcome-head` | The generative simulator's refreshed outcome head, "part of the shipped configuration" (§5.6) | 91,956 | `checkpoints/` |
| `cw-sim-test-vector-head` | The refreshed test-vector head, "what ships" (Appendix A.3) | 93,776 | `checkpoints/` |
| `actor` | The trained actor at the headline planner grid (§8.2, Figure 10) | 46,274 | `checkpoints/` |
| `actor-<belief>-seed<k>` | The trained actor re-run on three underlying beliefs, three seeds each (§8.2) | 78,594 each | `checkpoints/` |
| `sim-trained-actor` | The policy trained entirely inside the simulator, epoch-3 checkpoint (§8.3, Figure 11) | 78,594 | `checkpoints/` |
| `llm-selector-lora` | The language model fine-tuned on the simulator's pricing: a rank-16 adapter on the query and value projections of the frozen backbone (§8.4, Figure 12) | 4,538,368 | `checkpoints/` |
| `utility-model-seed<k>` | The utility model that prices a question from its meaning, three seeds (§8.4) | 59,777 each | `checkpoints/` |

The backbone, Gemma 4 E4B (7.94 B parameters), is frozen and is not redistributed. Download it from the Hugging Face Hub under the Gemma Terms of Use.

The adapter behind §8.4's "writing the question" experiment is not published, because it reproduces real CodingBat problem statements verbatim 29.7% of the time. Its training and evaluation code is included.

## Development names

The code refers to some models by the names used during development. They are kept inside the code so that recorded provenance still resolves.

| Release name | Name in the code |
|---|---|
| `cw-classifier` | `ratchet-original`; run `cw_sigma_precision_0305a22` |
| `cw-sim` (with its own outcome head) | `ratchet-cwsim/cold-head`; run `cw_sim_base_run` |
| `cw-sim` with `cw-sim-outcome-head` | `ratchet-cwsim/refreshed-head` |
| `cw-sim-test-vector-head` | `ratchet-cwsim/refreshed-tv-head` |
| `actor-classifier-*`, `actor-sim-cold-head-*`, `actor-sim-refreshed-head-*` | `results/rl_2/<substrate>/policy_seed*.pt` |

## Format and loading

Each model is a `.safetensors` file (tensors) plus a `.json` file (configuration and structure). `make weights` runs `scripts/materialize_checkpoints.py`, which rebuilds the exact checkpoint each script loads, writes it to the path the code expects, and checks it against `checkpoints/SHA256SUMS`. Several scripts verify a checkpoint's hash before using it, so use the torch version pinned in `uv.lock`.

Every published tensor is bit-identical to the training checkpoint. Optimizer and random-number-generator state were removed, and file paths in the stored configuration were made relative. `checkpoints/PROVENANCE.md` maps each published file to the training checkpoint recorded in the result files.

## Training data and intended use

- `fa-*` models were trained on FoundationalASSIST (CC-BY-NC-4.0, gated). `cw-*` models, the actors, the selector and the utility model were trained on CodeWorkout data.
- Evaluation is on learners held out of all training: 500 on mathematics, 110 on programming.
- **Intended use:** research on learner modelling, simulation and question selection. **Not** for commercial use, for decisions about real learners, or for any attempt to identify a learner.
- **Known limits:** stated in the thesis's Section 9.2. In particular, the belief's width tracks its errors less well than pre-registered (ratio 1.08 against a bar of 1.2). The model has no transition law, so nothing here is evidence that it can teach.

## Licence

Research and non-commercial use only, under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (the weights are Model Derivatives of Gemma 4 E4B) and the licences of the training data.
