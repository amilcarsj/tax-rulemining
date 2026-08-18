# Methodology Reconstruction

## Research Problem

The repository addresses unsupervised analysis of movement trajectory feature
datasets. The input is a vectorized trajectory-level CSV plus a taxonomy JSON
that organizes movement variables into interpretable feature groups.

The current method aims to convert unlabeled trajectories into interpretable
taxonomy-level behavior patterns:

1. Score each taxonomy node for local uncommonness using LOF.
2. Compare node-level uncommonness scores pairwise.
3. Convert score relations into symbolic pseudo-label fingerprints.
4. Mine high-level rules over these fingerprints.
5. Compress related rules into semantic meta-patterns.
6. Explain each meta-pattern using feature-level differences between target
   trajectories and interpretable contrast groups.

This is not a raw GPS preprocessing or visual analytics system in its current
pipeline. The main pipeline consumes trajectory-level feature tables. However,
the repository now also contains feature-generation code for rebuilding shared
spatio-temporal trajectory features from point-level CSVs.

## Step 1: Taxonomy-Guided LOF Scoring and Pseudo-Labels

Code evidence:

- `src/pipeline/step1.py`
- `src/taxonomy/parser.py`
- `src/taxonomy/validation.py`
- `src/taxonomy/pairs.py`
- `src/scoring/lof.py`
- `src/scoring/pseudo_labels.py`

### Taxonomy Parsing

The taxonomy JSON is parsed recursively. Each node receives:

- `node_id`: normalized from the node name.
- `node_name`: original name.
- `parent_id`.
- `depth`.
- `is_leaf`.
- declared feature list.
- descendant feature union.
- ancestor IDs.

Internal nodes get their feature set from the union of descendant features.
Nodes are validated against the dataset schema. Missing and non-numeric
features are logged to `pipeline_metadata.csv`. Nodes with no valid numeric
features are skipped.

### Valid Pair Generation

Valid taxonomy-node pairs are generated from usable nodes only. A pair is
excluded when:

- the two nodes are the same;
- one node is an ancestor of the other;
- the nodes share valid features.

Pairs are written to `taxonomy_pairs.csv`.

### LOF Scoring

For each usable taxonomy node, Step 1 computes LOF independently.

Preprocessing implemented in `src/scoring/lof.py`:

1. Select node feature subset.
2. Convert values to numeric.
3. Impute missing values with the feature median.
4. Drop constant features after imputation.
5. Scale with `sklearn.preprocessing.RobustScaler`.
6. Fit `sklearn.neighbors.LocalOutlierFactor`.

LOF parameters:

- `metric = "euclidean"`.
- `contamination = "auto"`.
- `novelty = False`.
- `n_neighbors = 0` if `n_samples <= 1`.
- `n_neighbors = n_samples - 1` if `n_samples <= 6`.
- otherwise `n_neighbors = min(20, max(5, floor(sqrt(n_samples))))`.

Raw LOF score:

```text
raw_lof = -negative_outlier_factor_
```

Percentile normalization:

```text
rank = scipy.stats.rankdata(raw_lof, method="average")
lof_percentile_score = (rank - 1) / (n_samples - 1)
```

The score is in `[0, 1]`, where larger values mean more locally uncommon within
that taxonomy node.

### Pairwise Pseudo-Labels

For each valid pair `(A, B)`, Step 1 compares the two LOF percentile scores.
Let `score_a` and `score_b` be the node scores.

The deterministic region assignment in `src/scoring/pseudo_labels.py` is:

```text
zone 0 if score_a < 0.5 and score_b < 0.5
zone 1 if score_a < 0.5 and score_b > 0.5 and score_a < score_b - 0.5
zone 2 if score_a > 0.5 and score_b < score_a - 0.5
zone 3 otherwise
```

Current labels are:

- `Normal A-B` for zone 0.
- `Uncommon B` for zone 1.
- `Uncommon A` for zone 2.
- `Hybrid A-B` for zone 3.

Step 1 writes both long and wide forms:

- `pair_pseudo_labels.csv`: one row per trajectory and pair.
- `trajectory_pseudo_labels.csv`: one row per trajectory, one column per pair.

## Step 2: High-Level Fingerprint Analysis and Rule Mining

Code evidence:

- `src/pipeline/step2.py`
- `src/pipeline/meta_patterns.py`

The current Step 2 is FP-growth association-rule mining over pseudo-label
transactions, not the original strict Jumping Emerging Pattern implementation
requested earlier. Current summaries explicitly report:

```text
mining_algorithm = fp_growth_association_rules
rule_type = fp_growth_association_rule
meta_pattern_type = node_uncommonness_count_compression
```

### Fingerprint Transactions

Step 2 reads `trajectory_pseudo_labels.csv`.

Labels are canonicalized:

- `Normal ...` becomes `Normal`.
- `Hybrid ...` becomes `Hybrid`.
- `Uncommon X` remains `Uncommon X`.

Each non-missing cell becomes an item:

```text
pair_id=label
```

Example:

```text
Speed_Dynamics__x__Turning_Dynamics=Uncommon Turning_Dynamics
```

### Pair Depth Filtering

Step 2 now requires `node_feature_sets.csv` next to the fingerprint file. It
keeps only same-depth taxonomy pairs. The current V2 reproduction commands
also restrict mining to depth 1:

```text
--min-node-depth 1
--max-node-depth 1
```

This prevents mining rules that mix internal and leaf levels of the taxonomy.

### Target and Antecedent Scope

Current defaults:

- targets are uncommon labels only;
- normal targets are excluded unless `--include-normal-targets` is used;
- hybrid items are excluded from antecedents by default;
- target-label echoes are excluded by default in the CLI, but current V2
  README commands use `--include-target-label-echoes`, so the current V2
  outputs have `target_label_echoes_excluded_from_antecedents = False`.

Target prevalence filtering:

```text
min_target_count = max(1, ceil(min_target_prevalence_ratio * n_total))
```

Only target items meeting this frequency are considered.

### FP-Growth Itemset Mining

Step 2 implements an FP-tree and mines frequent pseudo-label itemsets. The
support threshold is:

```text
min_support = max(min_support_count, ceil(min_support_ratio * n_total))
```

The maximum mined itemset length is:

```text
max_fp_growth_itemset_length = max_high_level_rule_length + 1
```

In current V2 runs, `max_high_level_rule_length = 2`, so itemsets up to length
3 are mined.

### Association Rule Metrics

For an antecedent `P` and target item `T`, the implementation computes:

```text
rule_support_count = count(P and T)
antecedent_support_count = count(P)
n_target = count(T)
n_contrast = n_total - n_target
support_contrast_count = count(P) - count(P and T)
confidence = count(P and T) / count(P)
target_base_rate = count(T) / n_total
lift = confidence / target_base_rate
target_coverage = count(P and T) / count(T)
rule_support_ratio = count(P and T) / n_total
antecedent_support_ratio = count(P) / n_total
leverage = support(P and T) - support(P) * support(T)
```

Growth rate is:

```text
support_target_ratio = count(P and T) / count(T)
support_contrast_ratio = support_contrast_count / n_contrast
growth_rate = support_target_ratio / support_contrast_ratio
```

If `support_contrast_ratio = 0`, the mathematical growth rate is infinite.
The current CSV export writes a blank `growth_rate` cell and records
`growth_rate_status = zero_contrast`. This avoids literal `inf` values while
preserving interpretability.

### Rule Filtering and Redundancy Pruning

Rules are retained only if they meet:

- minimum confidence;
- minimum lift;
- minimum target coverage;
- valid antecedent constraints.

Antecedent constraints:

- no item from the same pair column as the target;
- at most one item per pair column;
- optionally no repeated target label in the antecedent.

Rules are then pruned: a longer rule is dropped if an already accepted subset
rule with the same target has at least as high confidence and at least as high
lift.

### Behavioral Catalogue

`behavioral_pattern_catalogue.csv` turns retained rules into deterministic
plain-language text and includes support, target coverage, confidence, lift,
growth-rate status, and covered trajectory IDs.

### Semantic Meta-Pattern Compression

Step 2 compresses rules into semantic meta-patterns using uncommon-node counts.

For each rule, the antecedent items plus target item are transformed into node
counts:

```text
Uncommon Speed_Dynamics -> Speed_Dynamics += 1
Uncommon Curvature_Geometry -> Curvature_Geometry += 1
```

A semantic meta-pattern is a required-count signature such as:

```text
Curvature_Geometry>=2;Turning_Dynamics>=1
```

A trajectory satisfies the meta-pattern if its retained fingerprint has at
least those uncommon counts for the required nodes.

Meta-pattern importance:

```text
importance =
  mean_source_confidence
  * mean_source_lift
  * log1p(n_covered)
  * log1p(n_source_rules)
```

Outputs:

- `semantic_meta_patterns.csv`.
- `semantic_meta_pattern_coverage.csv`.
- `semantic_meta_pattern_source_rules.csv`.
- `semantic_behavioral_pattern_catalogue.csv`.

## Step 3: Semantic Meta-Pattern Explanation

Code evidence:

- `src/pipeline/step3.py`

Current Step 3 explains semantic meta-patterns by comparing trajectories that
satisfy each meta-pattern against three contrast groups.

### Contrast Groups

For each meta-pattern:

- `covered`: trajectories satisfying the required uncommon-count signature.
- `near_miss`: non-pattern trajectories closest to satisfying the meta-pattern,
  preferably one missing uncommonness count away.
- `matched_non_pattern`: non-pattern trajectories matched by total uncommon,
  hybrid, and pattern-evidence complexity.
- `typical_normal`: low-uncommonness reference trajectories, preferably zero
  uncommon and zero hybrid labels.

The older rule-level groups remain in code but are disabled by default:

- `rule_positive`.
- `antecedent_only`.
- `target_only`.
- `neither`.

### Comparison-Level Evidence

For each required node and retained pair containing that node, Step 3 computes
how often the node appears as `Uncommon node` in the target and contrast group.
It writes:

- covered count and ratio;
- contrast count and ratio;
- enrichment ratio with status for zero denominators.

Output:

- `meta_pattern_comparison_evidence.csv`.

### Feature-Level Evidence

For the features belonging to the required taxonomy nodes, Step 3 computes raw
and robust-standardized summaries for target and contrast groups.

Feature summary output:

- mean, median, standard deviation, q25, q75;
- mean and median robust-standardized values.

Feature difference output:

- raw mean and median differences;
- robust-standardized mean and median differences;
- absolute robust-standardized differences;
- direction labels.

Outputs:

- `meta_pattern_feature_summary.csv`.
- `meta_pattern_feature_differences.csv`.
- `meta_pattern_explanation_summary.csv`.

## Code-to-Concept Mapping

| Methodological concept | Code component |
| --- | --- |
| Dataset schema validation | `src/data_io/dataset.py` |
| Taxonomy parsing | `src/taxonomy/parser.py` |
| Missing/non-numeric feature validation | `src/taxonomy/validation.py` |
| Valid disjoint pair generation | `src/taxonomy/pairs.py` |
| Node-level LOF scoring | `src/scoring/lof.py` |
| Pseudo-label regions | `src/scoring/pseudo_labels.py` |
| Step 1 orchestration | `src/pipeline/step1.py` |
| Fingerprint summaries | `src/pipeline/step2.py` |
| FP-growth over pseudo-labels | `src/pipeline/step2.py` |
| Rule pruning and metrics | `src/pipeline/step2.py` |
| Semantic meta-pattern compression | `src/pipeline/step2.py`, `src/pipeline/meta_patterns.py` |
| Meta-pattern contrast explanations | `src/pipeline/step3.py` |
| Point-to-trajectory feature aggregation | `src/feature_engineering/spatiotemporal.py` |
| Flat LOF baseline | `src/baselines/flat_lof.py` |
| Raw feature association-rule baseline | `src/baselines/raw_association.py` |
| Uncompressed pseudo-label rule baseline | `src/baselines/pseudo_label_rules.py` |
| Read-only result explorer | `webapp/` |

