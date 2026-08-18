# Research Documentation Audit

This folder reconstructs the current state of the repository as an academic
research project. It is based on the source tree, README files, configuration,
generated outputs, and available Git metadata inspected on 2026-08-18.

No implementation files were modified for this audit.

## Important Caveat

The user requested the `research/` documentation described in `AGENTS.md`, but
no `AGENTS.md` file was found in the repository. Evidence: a repository-wide
search for `AGENTS.md` returned no files. The structure below is therefore a
best-effort research documentation structure.

## Contents

- `repository_audit.md`: repository inventory, configuration, tests,
  documentation, generated outputs, and Git status.
- `methodology.md`: reconstructed research problem, processing pipeline,
  algorithms, mathematical operations, and code-component mapping.
- `datasets_and_taxonomy.md`: dataset inventory, feature-generation code,
  taxonomy V1 and V2 structure, feature overlaps, and preprocessing evidence.
- `experiments_and_results.md`: current V2 experimental configuration and
  evidence-backed result summaries for fox, AIS, and hurricanes.
- `baselines_and_variants.md`: current baselines, legacy alternatives, removed
  or absent evaluation pieces, and ablation evidence.
- `paper_claims_evidence.md`: potential scientific claims and the exact
  repository evidence that supports, weakly supports, or does not yet support
  each claim.
- `limitations_and_questions.md`: limitations, unresolved questions, and
  researcher-input questions that cannot be answered from the repository.

## Current High-Level Reconstruction

The project implements a staged Python pipeline for unlabeled movement
trajectory feature tables. It uses a movement-feature taxonomy to compute
Local Outlier Factor scores per taxonomy node, turns pairwise node score
relations into pseudo-label fingerprints, mines high-level FP-growth
association rules over those fingerprints, compresses rules into semantic
meta-patterns, and explains those meta-patterns through taxonomy-organized
feature contrasts against interpretable contrast groups.

The current reproducible line of results is under:

- `output/fox/taxonomy_v2/`
- `output/ais/taxonomy_v2/`
- `output/hurricanes/taxonomy_v2/`

Legacy outputs also remain under `output/<dataset>/step1`,
`output/<dataset>/step2_association_refined`, and
`output/<dataset>/step3_counterexamples`.

