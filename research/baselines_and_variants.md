# Baselines, Ablations, and Variants

## Current Baseline Suite

The current baseline suite is implemented under `src/baselines/` and can be
run with:

```text
python -m baselines
```

The suite contains three comparison baselines or ablations.

## Baseline 1: Flat LOF

Code evidence:

- `src/baselines/flat_lof.py`

This baseline computes one global LOF score over all selected numeric features
instead of scoring each taxonomy node separately.

It reuses the same LOF strategy as Step 1:

- median imputation;
- constant-feature removal;
- RobustScaler;
- adaptive `n_neighbors`;
- LOF raw scores and percentile normalization.

Current outputs:

- `output/<dataset>/baselines/flat_lof_scores.csv`
- `output/<dataset>/baselines/flat_lof_feature_metadata.csv`
- `output/<dataset>/baselines/flat_lof_summary.json`

Current results:

| Dataset | Rows scored | Features used | n_neighbors |
| --- | ---: | ---: | ---: |
| fox | 1,551 | 100 | 20 |
| ais | 4,670 | 103 | 20 |
| hurricanes | 2,688 | 103 | 20 |

## Baseline 2: Raw Feature FP-Growth Association Rules

Code evidence:

- `src/baselines/raw_association.py`

This baseline ignores the taxonomy-guided LOF pseudo-label layer. It
discretizes raw numeric trajectory features into tertile bins and mines
association rules directly over feature-bin items.

Discretization:

- each usable numeric feature is split into low, medium, and high bins using
  33.33 and 66.67 percentiles over non-missing values;
- all-missing, constant, or insufficient-distinct-quantile features are
  skipped;
- item format is `feature=bin`.

Mining:

- custom FP-growth implementation;
- same-feature conflicts are disallowed by construction;
- current max raw itemset length is 2;
- current min support ratio is 0.10;
- current min confidence is 0.60;
- current min lift is 1.20.

Current results:

| Dataset | Discretized features | Frequent itemsets | Raw rules | Max itemset length | Output limit reached |
| --- | ---: | ---: | ---: | ---: | --- |
| fox | 100 | 33,079 | 3,433 | 2 | false |
| ais | 100 | 25,689 | 8,099 | 2 | false |
| hurricanes | 100 | 28,641 | 3,100 | 2 | false |

Evidence:

- `output/<dataset>/baselines/raw_association_summary.json`
- `output/baseline_comparison_summary.csv`

## Baseline 3: Pseudo-Label Rules Without Semantic Compression

Code evidence:

- `src/baselines/pseudo_label_rules.py`

This is best described as an ablation rather than an independent baseline. It
exports the Step 2 rule table before semantic meta-pattern compression.

Current results:

| Dataset | Uncompressed pseudo-label rules | Semantic meta-patterns | Rules per meta-pattern | Mean confidence | Mean lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| fox | 15 | 8 | 1.875 | 0.8908 | 8.0936 |
| ais | 293 | 30 | 9.7667 | 0.8610 | 7.5175 |
| hurricanes | 205 | 30 | 6.8333 | 0.8667 | 7.8219 |

Evidence:

- `output/<dataset>/baselines/pseudo_label_rules_without_compression_summary.json`
- `output/baseline_comparison_summary.csv`

## Legacy Variants Still Present in Outputs

Legacy output folders show prior variants:

- `output/<dataset>/step1`
- `output/<dataset>/step2_association_refined`
- `output/<dataset>/step3_counterexamples`

Legacy Step 2 summaries show:

| Dataset | Legacy pair columns retained | Legacy rule count | Legacy mining algorithm in summary |
| --- | ---: | ---: | --- |
| fox | 76 | 15 | `fp_growth_association_rules` |
| ais | 76 | 9 | `association_rules` |
| hurricanes | 76 | 7 | `association_rules` |

This indicates that the project evolved from a broader same-depth pair-mining
configuration toward the current V2 top-node-only mining configuration.

Legacy Step 3 summaries show older rule-level counterexample or nearest
counterexample outputs:

| Dataset | Legacy rules | Legacy neighbor/counterexample evidence |
| --- | ---: | --- |
| fox | 15 | 1,242 counterexample pairs; 63,659 feature comparisons |
| ais | 9 | 593 neighbor pairs; 22,887 feature comparisons |
| hurricanes | 7 | 283 neighbor pairs; 21,704 feature comparisons |

Current Step 3 still contains optional rule-level output code, but current V2
results disable it by default and focus on semantic meta-pattern contrast sets.

## Removed or Absent Evaluation Pieces

The repository does not currently contain implementation or outputs for:

- strict Jumping Emerging Pattern mining as originally specified;
- near-JEP mining;
- bootstrap rule stability;
- null-model label shuffling;
- sensitivity-analysis outputs.

Evidence:

- Source search did not find current bootstrap/null/JEP implementation.
- Current Step 2 summaries report `fp_growth_association_rules`, not JEP.
- No current V2 files such as `high_level_rule_stability.csv`,
  `high_level_rule_null_evaluation.csv`, or `null_model_summary.csv` were found.

## Algorithm Replacement Evidence

Current source code contains custom FP-growth implementations in both:

- `src/pipeline/step2.py`
- `src/baselines/raw_association.py`

Current source search did not find active Apriori or ECLAT implementations.
Legacy summaries for AIS and hurricanes use generic `association_rules`, but
without ordinary Git commits, the exact historical implementation cannot be
fully reconstructed from the repository alone.

## Baseline Interpretation for a Paper

The three current comparisons support three distinct paper questions:

1. Flat LOF asks whether taxonomy decomposition adds interpretability beyond a
   single global outlier score.
2. Raw feature FP-growth asks whether direct discretized-feature associations
   produce a large, low-level rule set compared with taxonomy-guided semantic
   patterns.
3. Uncompressed pseudo-label rules ask whether semantic meta-pattern
   compression makes the taxonomy-guided result set more compact without
   changing the underlying mined evidence.

NEEDS RESEARCHER INPUT: The paper should define what "better" means for these
baselines. Possible criteria include interpretability, rule-set compactness,
domain plausibility, coverage, stability, analyst workload, and runtime.

