# tax-rulemining

Python pipeline for taxonomy-guided movement feature analysis.

## Getting started

1. Activate the virtual environment:
   - macOS/Linux: `source .venv/bin/activate`
2. Install packaging support in the venv if needed:
   - `python -m pip install setuptools`
3. Install the project in editable mode:
   - `python -m pip install --no-build-isolation -e .`

## Current CLI

The project is wrapped as a staged CLI. Step 1 builds taxonomy-guided LOF
pseudo-label fingerprints, Step 2 mines and compresses high-level rules, and
Step 3 explains semantic meta-patterns through taxonomy-organized contrast
sets.

```bash
python tax_rulemining.py step1 \
  --data data/features.csv \
  --taxonomy data/taxonomy.json \
  --trajectory-id trajectory_id \
  --output-dir results/
```

For convenience, the legacy form still works and is treated as `step1`:

```bash
python tax_rulemining.py \
  --data data/features.csv \
  --taxonomy data/taxonomy.json \
  --trajectory-id trajectory_id \
  --output-dir results/
```

Step 1 outputs:

- `node_feature_sets.csv`
- `taxonomy_pairs.csv`
- `pipeline_metadata.csv`
- `node_lof_scores.csv`
- `pair_pseudo_labels.csv`
- `step1_summary.json`

## Step 2 Rule Focus

Step 2 mines rules with `Uncommon ...` labels as their targets. Hybrid labels
are excluded from rule antecedents by default, reducing the candidate search
space and keeping the resulting catalogue focused on uncommon behaviours. Use
`--include-hybrid-antecedents` only when Hybrid context is needed for a
comparison run. The default also excludes antecedents that repeat the target
uncommon label, avoiding redundant self-echo rules across overlapping pairs;
`--include-target-label-echoes` restores them for comparison. `--include-normal-targets`
additionally permits Normal targets. Association rules are filtered by joint
support, confidence, and lift; use `--min-confidence` and `--min-lift` to
tighten the catalogue further. `--min-target-coverage` requires a rule to
explain a minimum fraction of its target trajectories.

## Reproducing The Current V2 Results

Run the following from the project root after activating the virtual
environment. Each Step 2 command reads the fingerprint and taxonomy metadata
written by the preceding Step 1 command. Outputs are written to
`output/<dataset>/taxonomy_v2/step1/`,
`output/<dataset>/taxonomy_v2/step2_top_nodes/`, and
`output/<dataset>/taxonomy_v2/step3_meta_patterns/`.

```bash
# AIS
python tax_rulemining.py step1 \
  --data datasets/ais/ais-traj-feats.csv \
  --taxonomy datasets/ais/ais-taxonomy-v2.json \
  --trajectory-id trajectory_id \
  --output-dir output/ais/taxonomy_v2/step1

python tax_rulemining.py step2 \
  --fingerprint output/ais/taxonomy_v2/step1/trajectory_pseudo_labels.csv \
  --trajectory-id trajectory_id \
  --output-dir output/ais/taxonomy_v2/step2_top_nodes \
  --max-high-level-rule-length 2 \
  --min-support-count 20 \
  --min-support-ratio 0.0 \
  --min-confidence 0.80 \
  --min-lift 2.0 \
  --min-target-coverage 0.0 \
  --min-target-prevalence-ratio 0.02 \
  --min-node-depth 1 \
  --max-node-depth 1 \
  --include-target-label-echoes

python tax_rulemining.py step3 \
  --data datasets/ais/ais-traj-feats.csv \
  --step1-dir output/ais/taxonomy_v2/step1 \
  --step2-dir output/ais/taxonomy_v2/step2_top_nodes \
  --trajectory-id trajectory_id \
  --output-dir output/ais/taxonomy_v2/step3_meta_patterns

# Fox
python tax_rulemining.py step1 \
  --data datasets/fox/fox-traj-feats.csv \
  --taxonomy datasets/fox/fox-taxonomy-v2.json \
  --trajectory-id trajectory_id \
  --output-dir output/fox/taxonomy_v2/step1

python tax_rulemining.py step2 \
  --fingerprint output/fox/taxonomy_v2/step1/trajectory_pseudo_labels.csv \
  --trajectory-id trajectory_id \
  --output-dir output/fox/taxonomy_v2/step2_top_nodes \
  --max-high-level-rule-length 2 \
  --min-support-count 20 \
  --min-support-ratio 0.0 \
  --min-confidence 0.80 \
  --min-lift 2.0 \
  --min-target-coverage 0.0 \
  --min-target-prevalence-ratio 0.02 \
  --min-node-depth 1 \
  --max-node-depth 1 \
  --include-target-label-echoes

python tax_rulemining.py step3 \
  --data datasets/fox/fox-traj-feats.csv \
  --step1-dir output/fox/taxonomy_v2/step1 \
  --step2-dir output/fox/taxonomy_v2/step2_top_nodes \
  --trajectory-id trajectory_id \
  --output-dir output/fox/taxonomy_v2/step3_meta_patterns

# Hurricanes
python tax_rulemining.py step1 \
  --data datasets/hurricanes/hurricanes-traj-feats.csv \
  --taxonomy datasets/hurricanes/hurricanes-taxonomy-v2.json \
  --trajectory-id trajectory_id \
  --output-dir output/hurricanes/taxonomy_v2/step1

python tax_rulemining.py step2 \
  --fingerprint output/hurricanes/taxonomy_v2/step1/trajectory_pseudo_labels.csv \
  --trajectory-id trajectory_id \
  --output-dir output/hurricanes/taxonomy_v2/step2_top_nodes \
  --max-high-level-rule-length 2 \
  --min-support-count 20 \
  --min-support-ratio 0.0 \
  --min-confidence 0.80 \
  --min-lift 2.0 \
  --min-target-coverage 0.0 \
  --min-target-prevalence-ratio 0.02 \
  --min-node-depth 1 \
  --max-node-depth 1 \
  --include-target-label-echoes

python tax_rulemining.py step3 \
  --data datasets/hurricanes/hurricanes-traj-feats.csv \
  --step1-dir output/hurricanes/taxonomy_v2/step1 \
  --step2-dir output/hurricanes/taxonomy_v2/step2_top_nodes \
  --trajectory-id trajectory_id \
  --output-dir output/hurricanes/taxonomy_v2/step3_meta_patterns
```

The Step 2 rule table is `high_level_association_rules.csv`; the companion
`behavioral_pattern_catalogue.csv` provides the plain-language pattern and
evidence text for each retained rule.

## Step 3 Contrast Sets

Step 3 explains each semantic meta-pattern by comparing the target group
against three interpretable contrast sets:

- `near_miss`: trajectories that do not satisfy the meta-pattern but are
  closest to doing so, preferably one missing uncommonness count away.
- `matched_non_pattern`: non-pattern trajectories matched to the target group
  by overall uncommon-label and hybrid-label complexity.
- `typical_normal`: low-uncommonness reference trajectories, preferably with no
  uncommon and no hybrid labels across the retained top-node comparisons.

The main Step 3 files are:

- `meta_pattern_contrast_groups.csv`
- `meta_pattern_comparison_evidence.csv`
- `meta_pattern_feature_summary.csv`
- `meta_pattern_feature_differences.csv`
- `meta_pattern_explanation_summary.csv`
- `step3_summary.json`

## Local visual explorer

The separate `webapp/` folder contains a read-only Flask MVC application for
exploring the generated pipeline outputs. It provides a dataset overview,
filterable semantic meta-pattern catalogue, and taxonomy-organized feature
contrasts for the target group versus the three Step 3 contrast sets.

```bash
.venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/python webapp/run.py
```

Open <http://127.0.0.1:5000>. See [webapp/README.md](webapp/README.md) for
the output-folder assumptions and application details.
