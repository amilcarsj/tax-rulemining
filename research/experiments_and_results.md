# Experiments and Evidence-Backed Results

## Current Reproducible Configuration

The current V2 reproduction commands are documented in `README.md`. For each
dataset, the pipeline is:

1. Run Step 1 with `datasets/<dataset>/<dataset>-taxonomy-v2.json`.
2. Run Step 2 on the resulting `trajectory_pseudo_labels.csv`.
3. Run Step 3 on the Step 1 and Step 2 outputs.

Current V2 Step 2 settings from `step2_summary.json`:

| Parameter | Value |
| --- | --- |
| mining algorithm | `fp_growth_association_rules` |
| target scope | uncommon only |
| include normal targets | false |
| hybrid antecedents excluded | true |
| target-label echoes excluded | false |
| max antecedent length | 2 |
| max FP-growth itemset length | 3 |
| min support count | 20 |
| min support ratio | 0.0 |
| min confidence | 0.8 |
| min lift | 2.0 |
| min target coverage | 0.0 |
| min target prevalence ratio | 0.02 |
| same depth only | true |
| min node depth | 1 |
| max node depth | 1 |

Evidence:

- `output/fox/taxonomy_v2/step2_top_nodes/step2_summary.json`
- `output/ais/taxonomy_v2/step2_top_nodes/step2_summary.json`
- `output/hurricanes/taxonomy_v2/step2_top_nodes/step2_summary.json`

## Current Step 1 Results

| Dataset | Rows | Columns | Numeric features | Usable nodes | Valid pairs | LOF-scored nodes | Pseudo-label rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fox | 1,551 | 106 | 105 | 20 | 158 | 20 | 245,058 |
| ais | 4,670 | 106 | 105 | 20 | 158 | 20 | 737,860 |
| hurricanes | 2,688 | 106 | 105 | 20 | 158 | 20 | 424,704 |

All current V2 LOF runs used `n_neighbors = 20`, because all datasets are large
enough for the adaptive rule to hit the upper bound.

Evidence:

- `output/<dataset>/taxonomy_v2/step1/step1_summary.json`
- `output/<dataset>/taxonomy_v2/step1/node_lof_scores.csv`

## Current Step 2 Results

| Dataset | Original pair columns | Retained depth-1 pair columns | Target items | Rules | Semantic meta-patterns |
| --- | ---: | ---: | ---: | ---: | ---: |
| fox | 158 | 15 | 30 | 15 | 8 |
| ais | 158 | 15 | 30 | 293 | 30 |
| hurricanes | 158 | 15 | 30 | 205 | 30 |

Evidence:

- `output/<dataset>/taxonomy_v2/step2_top_nodes/step2_summary.json`
- `output/<dataset>/taxonomy_v2/step2_top_nodes/high_level_association_rules.csv`
- `output/<dataset>/taxonomy_v2/step2_top_nodes/semantic_meta_patterns.csv`

All current retained rules have antecedent length 2. This follows from the
current support, confidence, lift, and pruning configuration, not from an
explicit minimum antecedent length in Step 2.

## Pseudo-Label Composition for Retained Step 2 Pairs

| Dataset | Normal labels | Hybrid labels | Uncommon labels |
| --- | ---: | ---: | ---: |
| fox | 6,569 | 12,403 | 4,293 |
| ais | 18,340 | 35,718 | 15,992 |
| hurricanes | 10,827 | 20,678 | 8,815 |

Hybrid labels remain the largest class among retained depth-1 pair
observations. Current Step 2 excludes Hybrid items from antecedents and never
uses Hybrid targets by default, so the current rule tables focus on Uncommon
labels despite Hybrid prevalence.

Evidence:

- `output/<dataset>/taxonomy_v2/step2_top_nodes/pair_label_summary.csv`
- `src/pipeline/step2.py`

## Current Step 2 Example Rules

Examples from `output/fox/taxonomy_v2/step2_top_nodes/high_level_association_rules.csv`:

```text
R00001
IF Curvature_Geometry__x__Temporal_Sequential_Context=Uncommon Curvature_Geometry
AND Speed_Dynamics__x__Turning_Dynamics=Uncommon Turning_Dynamics
THEN Curvature_Geometry__x__Speed_Dynamics=Uncommon Curvature_Geometry
support=25, confidence=0.9615, lift=9.6216
```

```text
R00002
IF Acceleration_Dynamics__x__Turning_Dynamics=Uncommon Acceleration_Dynamics
AND Curvature_Geometry__x__Spatial_Reach_and_Route=Uncommon Curvature_Geometry
THEN Curvature_Geometry__x__Turning_Dynamics=Uncommon Curvature_Geometry
support=20, confidence=0.9524, lift=8.1610
```

Examples from `output/ais/taxonomy_v2/step2_top_nodes/high_level_association_rules.csv`:

```text
R00001
IF Acceleration_Dynamics__x__Speed_Dynamics=Uncommon Speed_Dynamics
AND Curvature_Geometry__x__Temporal_Sequential_Context=Uncommon Temporal_Sequential_Context
THEN Curvature_Geometry__x__Speed_Dynamics=Uncommon Speed_Dynamics
support=43, confidence=0.9773, lift=8.1937
```

Examples from `output/hurricanes/taxonomy_v2/step2_top_nodes/high_level_association_rules.csv`:

```text
R00001
IF Curvature_Geometry__x__Spatial_Reach_and_Route=Uncommon Curvature_Geometry
AND Temporal_Sequential_Context__x__Turning_Dynamics=Uncommon Temporal_Sequential_Context
THEN Spatial_Reach_and_Route__x__Temporal_Sequential_Context=Uncommon Temporal_Sequential_Context
support=20, confidence=1.0, lift=11.2941, growth_rate_status=zero_contrast
```

## Current Semantic Meta-Pattern Results

Top fox meta-pattern examples:

```text
MP00001
Curvature Geometry is uncommon in at least 2 retained comparisons
and Turning Dynamics is uncommon in at least 1 retained comparison
n_covered=69, n_source_rules=3
```

```text
MP00002
Acceleration Dynamics is uncommon in at least 1 retained comparison
and Curvature Geometry is uncommon in at least 2 retained comparisons
n_covered=58, n_source_rules=3
```

Top AIS meta-pattern examples:

```text
MP00001
Acceleration Dynamics is uncommon in at least 2 retained comparisons
and Curvature Geometry is uncommon in at least 1 retained comparison
n_covered=228, n_source_rules=11
```

Top hurricanes meta-pattern examples:

```text
MP00001
Acceleration Dynamics is uncommon in at least 2 retained comparisons
and Speed Dynamics is uncommon in at least 1 retained comparison
n_covered=123, n_source_rules=11
```

Evidence:

- `output/<dataset>/taxonomy_v2/step2_top_nodes/semantic_meta_patterns.csv`

## Current Step 3 Results

| Dataset | Meta-patterns explained | Contrast-group rows | Comparison evidence rows | Feature summary rows | Feature difference rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| fox | 8 | 5,737 | 240 | 1,136 | 852 |
| ais | 30 | 59,965 | 900 | 4,160 | 3,120 |
| hurricanes | 30 | 34,442 | 900 | 4,160 | 3,120 |

Current Step 3 ran with `include_rule_level_outputs = false`, so rule-level
counterexample outputs are not part of the current V2 evidence line.

Evidence:

- `output/<dataset>/taxonomy_v2/step3_meta_patterns/step3_summary.json`
- `output/<dataset>/taxonomy_v2/step3_meta_patterns/meta_pattern_*`

## Interpretation of Blank Growth Rates

Blank `growth_rate` values in current rule CSVs are not missing computations.
They represent mathematically infinite growth rates caused by zero contrast
support. The companion field `growth_rate_status` records `zero_contrast`.

Evidence:

- `src/pipeline/step2.py`
- `output/hurricanes/taxonomy_v2/step2_top_nodes/high_level_association_rules.csv`

## Claims Not Yet Supported by Results

No external ground-truth classification, clustering quality, or domain expert
validation results were found.

No bootstrap stability or null-model evaluation outputs were found in current
V2 results.

No runtime benchmarks or scalability experiments were found, although user
experience during development indicated that unconstrained itemset mining could
be slow. Repository evidence for this is indirect only: the current code has
prevalence filtering, max antecedent length, same-depth filtering, and
FP-growth implementations.

