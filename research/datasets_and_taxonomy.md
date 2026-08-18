# Datasets, Features, and Taxonomy

## Dataset Inventory

The repository includes three bundled datasets:

| Dataset | Trajectory feature rows | Trajectory feature columns | Point feature rows | Point feature columns | GeoJSON files |
| --- | ---: | ---: | ---: | ---: | ---: |
| fox | 1,551 | 106 | 86,856 | 10 | 5,077 |
| ais | 4,670 | 106 | 373,600 | 10 | 8,207 |
| hurricanes | 2,688 | 106 | 150,528 | 10 | 2,688 |

Evidence:

- `datasets/fox/fox-traj-feats.csv`
- `datasets/ais/ais-traj-feats.csv`
- `datasets/hurricanes/hurricanes-traj-feats.csv`
- corresponding `*-point-feats.csv`
- `datasets/<dataset>/geojson/`

Point feature columns are:

```text
trajectory_id, object_id, time, lat, lon, label, distance, speed, acceleration, angle
```

Trajectory feature tables begin with:

```text
trajectory_id, object_id, distance_geometry_1_1, distance_geometry_2_1, ...
```

The current pipeline uses `trajectory_id` as the trajectory identifier and
excludes `object_id` from baseline numeric feature sets by default.

## Dataset Provenance

UNKNOWN. The repository does not document where the fox, AIS, or hurricanes
datasets came from, what labels mean, whether the data are public, what
coordinate reference system or sampling procedure was used, or what use
restrictions apply.

The existing `*-outlier-scores.csv` files are present but are not used by the
current V2 pipeline. They appear to correspond to older score columns:

```text
geometric, kinematic, curvature, indentation, speed, acceleration
```

The hurricanes outlier-score file has 2,687 rows while the current trajectory
feature table has 2,688 rows. This mismatch is not relevant to current V2 runs
unless those legacy scores are reused later.

## Feature Engineering Code

The repository includes repeatable feature-generation code:

- `src/feature_engineering/spatiotemporal.py`
- `src/feature_engineering/build_datasets.py`

`build_datasets.py` targets all three bundled datasets and rewrites
`datasets/<dataset>/<dataset>-traj-feats.csv` from the point-level file plus
existing trajectory table.

Feature families generated from point-level data include:

- temporal support: point count, total duration, sampling gap mean, standard
  deviation, interquartile range, and coefficient of variation;
- phase summaries: first-half and second-half mean speed and acceleration, and
  half differences;
- spatial footprint: latitude/longitude spans, bounding-box height, width,
  diagonal, and area;
- route/displacement: path length, net displacement, directness ratio;
- localization: mean, median, max distance to centroid, radius of gyration;
- evolution: speed, acceleration, and angle progress slopes;
- relative spatial sequence complexity: 3x3 relative grid unique cells,
  entropy, transition entropy, and revisit ratio.

Mathematical operations implemented include:

- haversine distance in meters;
- arithmetic mean;
- population standard deviation;
- median;
- linear-interpolated quantile;
- interquartile range;
- least-squares slope over normalized trajectory progress;
- Shannon entropy with base-2 logarithm;
- transition entropy over adjacent relative-grid states.

NEEDS RESEARCHER INPUT: It is not documented which trajectory features existed
before this feature-generation code and which were newly appended by it.

## Taxonomy V1

Each dataset has an original taxonomy JSON:

- `datasets/fox/fox-taxonomy.json`
- `datasets/ais/ais-taxonomy.json`
- `datasets/hurricanes/hurricanes-taxonomy.json`

V1 structure:

- 18 nodes.
- 12 leaves.
- root: `Movement variables`.
- depth-1 groups: `Kinematic`, `Geometric`, `Temporal`, `Spatial_Context`,
  `SpatioTemporal_Dynamics`.

V1 leaf groups include:

- `Speed`
- `Acceleration`
- `Curvature`
- `Indentation`
- `Duration`
- `Sampling_Gaps`
- `Phase_Change`
- `Extent`
- `Displacement`
- `Localization`
- `Path_Evolution`
- `Sequence_Complexity`

Legacy V1 results exist under `output/<dataset>/step1`,
`output/<dataset>/step2_association_refined`, and
`output/<dataset>/step3_counterexamples`.

## Taxonomy V2

Each dataset also has a V2 taxonomy JSON:

- `datasets/fox/fox-taxonomy-v2.json`
- `datasets/ais/ais-taxonomy-v2.json`
- `datasets/hurricanes/hurricanes-taxonomy-v2.json`

V2 structure:

- 20 nodes.
- 13 leaves.
- root: `Movement variables`.
- six depth-1 groups.

V2 depth-1 groups and valid feature counts in current outputs:

| Depth-1 node | Valid features |
| --- | ---: |
| Speed_Dynamics | 19 |
| Acceleration_Dynamics | 19 |
| Turning_Dynamics | 19 |
| Curvature_Geometry | 15 |
| Spatial_Reach_and_Route | 13 |
| Temporal_Sequential_Context | 19 |

V2 depth-2 leaves and valid feature counts:

| Leaf node | Parent | Valid features |
| --- | --- | ---: |
| Speed_Level_Profile | Speed_Dynamics | 10 |
| Speed_Variability_Shape | Speed_Dynamics | 9 |
| Acceleration_Level_Profile | Acceleration_Dynamics | 10 |
| Acceleration_Variability_Shape | Acceleration_Dynamics | 9 |
| Turning_Level_Profile | Turning_Dynamics | 10 |
| Turning_Variability_Shape | Turning_Dynamics | 9 |
| Curvature_Coarse_Multiscale | Curvature_Geometry | 6 |
| Curvature_Fine_Multiscale | Curvature_Geometry | 9 |
| Spatial_Footprint | Spatial_Reach_and_Route | 10 |
| Route_Efficiency | Spatial_Reach_and_Route | 3 |
| Observation_Temporal_Support | Temporal_Sequential_Context | 6 |
| Kinematic_Evolution | Temporal_Sequential_Context | 8 |
| Spatial_Sequence_Complexity | Temporal_Sequential_Context | 5 |

The V2 taxonomy uses the same structure and feature assignments across all
three datasets.

## Feature Overlap Audit

For V2 current Step 1 outputs:

- depth-1 taxonomy nodes have zero pairwise feature overlaps;
- generated valid taxonomy pairs have zero feature overlaps;
- parent-child feature overlap exists by design, because internal-node feature
  sets are unions of descendant features.

Evidence: `output/<dataset>/taxonomy_v2/step1/node_feature_sets.csv` and
`taxonomy_pairs.csv` for fox, AIS, and hurricanes.

This supports the current decision to mine only same-depth depth-1 pairs in
Step 2. It avoids mixing a parent node with child nodes in the same rule-mining
space.

## Step 1 Metadata Events

Current V2 metadata events:

| Dataset | Metadata events | Categories |
| --- | ---: | --- |
| fox | 12 | constant features |
| ais | 3 | constant features |
| hurricanes | 3 | constant features |

Fox constant-feature examples include `acceleration_0s`, `angles_0s`,
`point_count`, and `speed_0s`. AIS and hurricanes constant events include
`point_count` for root and temporal-support nodes.

Evidence:

- `output/fox/taxonomy_v2/step1/pipeline_metadata.csv`
- `output/ais/taxonomy_v2/step1/pipeline_metadata.csv`
- `output/hurricanes/taxonomy_v2/step1/pipeline_metadata.csv`

## Dataset and Taxonomy Unknowns

- Dataset provenance is UNKNOWN.
- Ground-truth labels, if any, are UNKNOWN.
- The scientific rationale for every V2 feature placement needs researcher
  validation.
- The reason for the hurricanes outlier-score row mismatch is UNKNOWN.
- The relationship between GeoJSON files and trajectory feature rows is
  partially unclear. Counts do not match for fox and AIS.

