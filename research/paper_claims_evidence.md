# Potential Paper Claims and Repository Evidence

This file separates possible scientific claims from repository evidence. It is
intentionally conservative: if a claim cannot be established from code or
outputs, it is marked as UNKNOWN or NEEDS RESEARCHER INPUT.

## Supported by Current Repository Evidence

### Claim: The tool implements a taxonomy-guided unsupervised pipeline for movement trajectory feature tables.

Evidence:

- `README.md` describes Step 1, Step 2, and Step 3.
- `src/pipeline/step1.py`, `src/pipeline/step2.py`, and `src/pipeline/step3.py`
  implement the staged pipeline.
- Current V2 outputs exist under `output/<dataset>/taxonomy_v2/`.

### Claim: The method computes independent LOF uncommonness scores for each usable taxonomy node.

Evidence:

- `src/scoring/lof.py` computes node-level LOF and writes rows by node.
- `output/<dataset>/taxonomy_v2/step1/node_lof_scores.csv` exists for all
  three datasets.
- Current V2 Step 1 summaries report 20 LOF-scored nodes for each dataset.

### Claim: The method turns pairwise taxonomy-node LOF scores into pseudo-label fingerprints.

Evidence:

- `src/scoring/pseudo_labels.py` implements deterministic pseudo-label regions.
- `src/pipeline/step1.py` writes `pair_pseudo_labels.csv` and
  `trajectory_pseudo_labels.csv`.
- Current V2 Step 1 summaries report pseudo-label row counts for all datasets.

### Claim: Current rule mining uses FP-growth association rules over pseudo-label transactions.

Evidence:

- `src/pipeline/step2.py` implements an FP-tree and
  `mine_frequent_pseudo_label_itemsets`.
- Current Step 2 summaries report `mining_algorithm =
  fp_growth_association_rules`.
- Current rule outputs are named `high_level_association_rules.csv`.

### Claim: Current Step 2 restricts mining to same-depth taxonomy pairs and current V2 experiments use only depth-1 pairs.

Evidence:

- `src/pipeline/step2.py` loads `node_feature_sets.csv` and filters to
  same-depth pairs.
- Current V2 Step 2 summaries report `same_depth_only = true`,
  `min_node_depth = 1`, and `max_node_depth = 1`.
- Current V2 Step 2 summaries report 15 retained pair columns out of 158.

### Claim: Semantic meta-patterns compress rule-level evidence into node-level uncommonness-count patterns.

Evidence:

- `src/pipeline/meta_patterns.py` implements uncommon-node counts and
  required-count satisfaction.
- `src/pipeline/step2.py` writes `semantic_meta_patterns.csv`,
  `semantic_meta_pattern_source_rules.csv`, and
  `semantic_meta_pattern_coverage.csv`.
- Current outputs show fox 15 rules compressed into 8 meta-patterns, AIS 293
  rules into 30 meta-patterns, and hurricanes 205 rules into 30 meta-patterns.

### Claim: Current Step 3 explains semantic meta-patterns with contrast groups rather than individual trajectory visualizations.

Evidence:

- `src/pipeline/step3.py` defines `near_miss`, `matched_non_pattern`, and
  `typical_normal` contrast groups.
- `output/<dataset>/taxonomy_v2/step3_meta_patterns/meta_pattern_*` files
  exist for all three datasets.
- `webapp/README.md` states that the interface does not show individual GPS
  tracks.

### Claim: The current V2 taxonomy has disjoint depth-1 feature groups.

Evidence:

- `output/<dataset>/taxonomy_v2/step1/node_feature_sets.csv` shows depth-1
  feature sets.
- An overlap audit over the output files found zero pairwise feature overlaps
  among depth-1 nodes for fox, AIS, and hurricanes.
- `output/<dataset>/taxonomy_v2/step1/taxonomy_pairs.csv` contains only valid
  non-overlapping feature pairs.

### Claim: Hybrid pseudo-labels are frequent, but current rule mining focuses on Uncommon labels.

Evidence:

- `output/<dataset>/taxonomy_v2/step2_top_nodes/pair_label_summary.csv` shows
  Hybrid counts larger than Normal and Uncommon counts for retained depth-1
  pair observations.
- `src/pipeline/step2.py` excludes Hybrid antecedents by default and never
  allows Hybrid targets.
- Current Step 2 summaries report `target_label_scope = uncommon_only`.

## Weakly Supported or Partially Supported Claims

### Claim: The taxonomy V2 was introduced to improve conceptual balance and cross-dataset relevance.

Evidence:

- V2 taxonomy files exist for all datasets and share the same six top-level
  groups.
- V2 outputs exist for all datasets.
- Codex tree diffs suggest V2 files were added after earlier outputs.

Limitation:

- No ordinary Git commits or design document explains the rationale.
- The exact researcher reasoning is not preserved in repository files.

### Claim: FP-growth replaced earlier Apriori/ECLAT-style mining for scalability.

Evidence:

- Current source uses FP-growth.
- Legacy Step 2 summaries for AIS and hurricanes report
  `mining_algorithm = association_rules`.
- Current source search did not find active Apriori or ECLAT implementations.

Limitation:

- No commit history documents the replacement.
- The previous algorithm implementation cannot be reconstructed completely from
  current source alone.

### Claim: Semantic compression improves interpretability.

Evidence:

- The compression substantially reduces rule tables for AIS and hurricanes:
  AIS 293 rules to 30 meta-patterns; hurricanes 205 rules to 30 meta-patterns.
- The webapp and Step 3 are organized around semantic meta-patterns.

Limitation:

- Interpretability was not evaluated with users or domain experts in the
  repository.
- "Improves interpretability" remains a paper hypothesis unless validated.

## Not Supported Yet

### Claim: The method outperforms baselines quantitatively.

Status: NOT ESTABLISHED.

Evidence exists for baseline outputs, but no accepted quantitative criterion is
documented. There is no ground-truth accuracy, expert judgment, stability test,
or runtime benchmark comparing methods.

### Claim: The mined patterns are scientifically meaningful for fox, AIS, or hurricane movement behavior.

Status: NEEDS RESEARCHER INPUT.

The repository shows patterns and feature differences, but does not contain
domain interpretation or validation.

### Claim: The method is robust or stable.

Status: NOT ESTABLISHED.

No current bootstrap stability or null-model evaluation outputs were found.

### Claim: The pseudo-label regions are theoretically optimal.

Status: NOT ESTABLISHED.

The deterministic region definitions are implemented, but the repository does
not provide a theoretical justification, sensitivity analysis, or comparison
against alternative region definitions.

### Claim: The datasets are representative of broader movement behavior.

Status: UNKNOWN.

Dataset provenance, sampling protocol, and selection criteria are not
documented.

## Evidence Trail for Key Result Numbers

Current V2 result evidence:

- `output/fox/taxonomy_v2/step1/step1_summary.json`
- `output/fox/taxonomy_v2/step2_top_nodes/step2_summary.json`
- `output/fox/taxonomy_v2/step3_meta_patterns/step3_summary.json`
- corresponding files for `ais` and `hurricanes`.

Baseline result evidence:

- `output/baseline_comparison_summary.csv`
- `output/<dataset>/baselines/baseline_summary.json`
- `output/<dataset>/baselines/raw_association_summary.json`
- `output/<dataset>/baselines/flat_lof_summary.json`
- `output/<dataset>/baselines/pseudo_label_rules_without_compression_summary.json`

Legacy variant evidence:

- `output/<dataset>/step2_association_refined/step2_summary.json`
- `output/<dataset>/step3_counterexamples/step3_summary.json`

