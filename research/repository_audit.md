# Repository Audit

## Source Tree

Core implementation is under `src/` and is organized by concept:

- `src/cli/main.py`: command-line interface for `step1`, `step2`, and `step3`.
- `src/core/`: shared dataclasses and progress reporting.
- `src/data_io/`: CSV and JSON loading/writing helpers.
- `src/taxonomy/`: taxonomy parsing, validation, and valid pair generation.
- `src/scoring/`: LOF scoring and pairwise pseudo-label generation.
- `src/pipeline/step1.py`: taxonomy parsing, LOF scoring, and fingerprint output.
- `src/pipeline/step2.py`: fingerprint summaries, FP-growth rule mining, rule
  pruning, behavioral catalogue, and semantic meta-pattern compression.
- `src/pipeline/meta_patterns.py`: shared helpers for semantic meta-pattern
  required-count logic.
- `src/pipeline/step3.py`: semantic meta-pattern explanation and optional
  legacy rule-level contrast/counterexample outputs.
- `src/feature_engineering/`: point-to-trajectory spatio-temporal feature
  aggregation code.
- `src/baselines/`: flat LOF, raw feature FP-growth association rules, and
  pseudo-label rules without semantic compression.

The repository-level runner is `tax_rulemining.py`. It injects `src/` into
`sys.path` and dispatches to `cli.main:main`.

## Documentation

Existing documentation before this audit:

- `README.md`: describes the staged CLI, current V2 reproduction commands,
  Step 2 rule focus, Step 3 contrast sets, and local webapp startup.
- `webapp/README.md`: describes the read-only Flask explorer and the expected
  V2 output paths.

No other Markdown documentation was found. No `AGENTS.md` was found.

## Configuration

The only project configuration file found was `pyproject.toml`.

Evidence from `pyproject.toml`:

- Project name: `tax-rulemining`.
- Version: `0.1.0`.
- Python requirement: `>=3.12`.
- Dependencies: `Flask>=3.0`, `numpy>=1.26`, `scipy>=1.11`,
  `scikit-learn>=1.4`.
- Console script: `tax-rulemining = "cli.main:main"`.
- Build backend: `setuptools.build_meta`.

The `.gitignore` excludes common Python generated artifacts, virtual
environment folders, coverage output, caches, and `.DS_Store`.

## Notebooks, Tests, and Experiment Scripts

No notebooks were found.

No test directories or test files were found. There is no `pytest`, `unittest`,
or other test configuration in the repository.

Experiment execution is currently documented through README command blocks,
not through standalone shell scripts or workflow files. The source tree also
contains `src/feature_engineering/build_datasets.py` for regenerating shared
trajectory features and `python -m baselines` for baseline runs.

## Generated Result Files

Current V2 outputs exist for all three datasets:

- `output/fox/taxonomy_v2/step1`, `step2_top_nodes`, `step3_meta_patterns`.
- `output/ais/taxonomy_v2/step1`, `step2_top_nodes`, `step3_meta_patterns`.
- `output/hurricanes/taxonomy_v2/step1`, `step2_top_nodes`,
  `step3_meta_patterns`.

Legacy outputs also remain:

- `output/<dataset>/step1`.
- `output/<dataset>/step2_association_refined`.
- `output/<dataset>/step3_counterexamples`.

Baseline outputs exist for all three datasets:

- `output/<dataset>/baselines/flat_lof_scores.csv`.
- `output/<dataset>/baselines/raw_association_rules.csv`.
- `output/<dataset>/baselines/pseudo_label_rules_without_compression.csv`.
- `output/baseline_comparison_summary.csv`.

## Web Application

The `webapp/` directory contains a read-only Flask MVC app. Evidence:
`webapp/app/models/repository.py` reads from:

- `output/<dataset>/taxonomy_v2/step1`.
- `output/<dataset>/taxonomy_v2/step2_top_nodes`.
- `output/<dataset>/taxonomy_v2/step3_meta_patterns`.

It exposes dataset dashboards, semantic meta-pattern lists, and
taxonomy-organized feature differences. It intentionally does not show
individual GPS tracks or rerun the pipeline, according to `webapp/README.md`.

## Git Status and History

The repository has an initialized Git directory, but there are no ordinary
commits on `main`.

Evidence:

- `git status --short --branch` reported `## No commits yet on main`.
- All repository files were untracked at audit time.
- `git log --oneline --decorate --graph --all --max-count=50` returned no
  ordinary commit history.
- No branches or tags were reported by normal branch/tag commands.

There are Codex tree refs under `refs/codex/...`. They are tree objects rather
than commits, so they provide weak historical evidence only. A tree diff
between available snapshots suggests movement toward:

- adding `*-taxonomy-v2.json`;
- adding V2 outputs under `output/<dataset>/taxonomy_v2/`;
- adding the baseline package under `src/baselines/`;
- adding `src/pipeline/meta_patterns.py`;
- modifying `step2.py`, `step3.py`, and the webapp;
- replacing old rule detail/list templates with semantic meta-pattern pages.

Because these are not normal commits, they should not be cited as reproducible
version history in a paper.

## Audit Gaps

- There is no formal commit history documenting why decisions changed.
- There are no tests to establish behavioral correctness.
- There are no notebooks or scripts capturing exploratory analyses.
- Dataset provenance and licensing are not documented in the repository.
- `AGENTS.md` is absent, so the requested documentation convention could not
  be followed literally.

