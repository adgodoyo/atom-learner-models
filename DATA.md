# Data

This repository contains **no data**: no interaction rows, no problem text, no learner answers, no learner code, and no per-learner records of any kind, including model outputs such as beliefs. Every corpus must be obtained from its owner under the owner's terms.

What it does contain, so that the published code runs and the splits can be checked:

- **Per-problem vectors and labels:** the frozen codec transform (`transforms/`), one encoder vector per problem (`artifacts/*codec*.npz`), each problem's format (`artifacts/fa_problem_type_map.csv`, `splits/fa_problem_groups_v2.csv`) and each CodeWorkout problem's CodingBat method name (`artifacts/cw_problem_method_names.csv`). None of them is invertible to the problem's text.
- **The frozen splits** (`splits/`): which learner and which problem belongs to which partition, and nothing else about them. Learners appear as `sha1(dataset learner id)[:16]`, unsalted, so that anyone who holds the dataset can recompute them and check the split; without the dataset they identify no one, and the datasets' own identifiers are already anonymous.
- **Aggregate results** (`results/`): summary statistics behind the thesis numbers, with no per-learner rows.

The code reads the corpora from a folder named `Datasets` placed **next to** the repository:

```
parent-folder/
├── atom-learner-models/          this repository
└── Datasets/
    ├── FoundationalAssist/       Data/Interactions.csv, Data/Problems.csv, …
    ├── CodeWorkout_CMU/          S19_All_Release_2_10_22.zip, F19_Release_All_05_23_22.zip (unopened)
    └── tiktoc-main/              the TIKTOC release (data/, evaluator/, test-case-query-results/, …)
```

The loaders also accept explicit locations: `ATOM_FA_DIR` (FoundationalASSIST), `ATOM_CW_RAW_DIR` (the CodeWorkout zips) and `ATOM_CW_DIR` (TIKTOC's `data/` folder).

## FoundationalASSIST (mathematics)

- **What:** 1,753,384 interactions of 5,000 learners with 3,395 grade 6–8 mathematics problems from the Illustrative Mathematics curriculum, recorded on ASSISTments, with the full problem text and each learner's first typed answer. After deduplication and the item split, the thesis uses 1,716,839 timed events over 3,323 problems.
- **Source:** Hugging Face, [`ASSISTments/FoundationalASSIST`](https://huggingface.co/datasets/ASSISTments/FoundationalASSIST). Access is gated: you agree to non-commercial use, to the ASSISTments Responsible Use Guidelines, and to secure storage without redistribution.
- **Licence:** CC-BY-NC-4.0 plus the access agreement.
- **Cite:** Worden et al. (2026), the FoundationalASSIST release paper.
- **Build:** `make data-fa` runs `scripts/fa_00_manifest_rebuild.py`. It deduplicates the release, builds the clean event stream, and checks the learner and item splits against `splits/fa_lockbox_manifest_v2.json`. The learner identifiers in `splits/` are the hashes described above.

## CodeWorkout (introductory Java)

- **What:** the Spring 2019 (413 learners) and Fall 2019 (506 learners) CodeWorkout cohorts on a shared catalogue of 50 Java problems: 195,205 program-run events in ProgSnap2 format, each with the source snapshot, compile result, score and timestamp, collected on Virginia Tech's CodeWorkout platform.
- **Source:** PSLC DataShop, project *Virginia Tech CodeWorkout*, [dataset 3458](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=3458). Create a DataShop account, accept the [project terms](https://pslcdatashop.web.cmu.edu/ProjectTerms?id=585) and download the release zips into `Datasets/CodeWorkout_CMU/`. The code expects `S19_All_Release_2_10_22.zip` and `F19_Release_All_05_23_22.zip`; the learner split also reads `F19_Release_Train_06-28-21.zip` and `F19_Release_Test_06-28-21.zip`.
- **Learner split:** the Fall 2019 train and test files are the ones released for the 2nd CSEDM Data Challenge (Educational Data Mining in Computer Science Education workshop); the thesis adopts that challenge's own train/test division of the Fall 2019 learners rather than drawing a new one.
- **Terms:** the [DataShop Terms of Use](https://pslcdatashop.web.cmu.edu/ProjectTerms?id=585): research use only, no attempt to de-anonymise, and no redistribution. Anyone who needs the data must create their own DataShop account and accept the terms, which is why no learner-level data is in this repository.
- **Build:** `make data-cw` runs `scripts/cw_1_event_graph.py`, `scripts/cw_2_sessions.py` and `scripts/cw_4c_step0_1_splits.py`, and checks the splits in `splits/`. The seven problems held out of all training are listed in `splits/cw_s19_item_split.csv`.

## CodingBat problem statements

The 50 CodeWorkout problems are CodingBat exercises (Nick Parlante, codingbat.com). Their statements are not redistributed here. `artifacts/cw_coding_bat_sources.json` holds each problem's method name, public URL and the SHA-256 of the statement the thesis used. Rebuild the statement file with:

```bash
python scripts/fetch_codingbat_statements.py      # writes artifacts/cw_coding_bat_recovery_raw.json, checks every hash
python scripts/cw_4b_recover_problem_statements.py
```

## TIKTOC (17-problem benchmark, test-case results, CodeBLEU)

- **What:** the TIKTOC authors' release: the 17 problems with recovered test suites, per-test-case results, and the CodeBLEU evaluator used by the CodeWorkout generation numbers.
- **Source:** [github.com/umass-ml4ed/tiktoc](https://github.com/umass-ml4ed/tiktoc), the release accompanying Duan et al., [arXiv:2410.10829](https://arxiv.org/abs/2410.10829). The thesis used commit `8d5b8747a06bb5af3974818b6c2711a7fffddc5a`. Clone it as `Datasets/tiktoc-main`; the loader unpacks the two archives in `data/` on first use:

  ```bash
  git clone https://github.com/umass-ml4ed/tiktoc ../Datasets/tiktoc-main
  git -C ../Datasets/tiktoc-main checkout 8d5b8747a06bb5af3974818b6c2711a7fffddc5a
  ```
- **Note:** CodeBLEU uses the release's compiled tree-sitter parser (`evaluator/CodeBLEU/parser/my-languages.so`), which is a Linux x86-64 binary. CodeBLEU numbers therefore need Linux; everything else runs on macOS too.

## XES3G5M

`configs/data.yaml` still has an entry for XES3G5M, a third corpus examined during development and not used in the thesis. No published script reads it, and nothing from it is in this repository.

## The backbone

Gemma 4 E4B, from the Hugging Face Hub under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Accept the terms on its model page and log in with `huggingface-cli login` before running anything that loads the backbone. It is frozen everywhere in this repository.
