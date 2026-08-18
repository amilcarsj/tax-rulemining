# Limitations and Researcher Questions

## Technical Limitations

- No tests exist. There is no automated regression suite for taxonomy parsing,
  LOF scoring, pseudo-label assignment, FP-growth mining, semantic compression,
  or Step 3 contrast generation.
- There is no normal Git commit history. Most files are untracked, and the
  only historical clues are Codex tree refs.
- The repository includes many generated outputs and data files, but no
  reproducibility manifest with commands, environment hashes, runtime, or
  random seeds.
- The current FP-growth code is custom. It should be tested carefully against a
  known implementation or small hand-checked examples.
- The CLI Step 2 description still mentions "quantitative rule evaluation",
  but current source and outputs do not contain bootstrap or null-model
  evaluation.
- Rule-level counterexample code remains in `src/pipeline/step3.py`, but
  current V2 outputs disable it by default and remove old rule-level files.
  This can confuse readers unless documented clearly.

## Methodological Limitations

- Pseudo-label regions are deterministic and threshold-based. The repository
  does not justify why `0.5` and the diagonal margin of `0.5` are appropriate.
- Hybrid labels are frequent. Current mining excludes Hybrid items from
  antecedents and targets by default, so the current paper story should explain
  why the focus is uncommonness rather than hybrid behavior.
- Same-depth depth-1 mining improves interpretability but discards possible
  leaf-level or cross-level relationships.
- Semantic meta-pattern compression may hide important rule-level distinctions.
  It should be presented as a summarization layer, not a replacement for all
  detailed evidence.
- Current baselines show output counts and rule metrics, but they do not yet
  establish scientific superiority.
- The raw association baseline currently uses max itemset length 2, which may
  limit multi-feature rule complexity.
- Step 3 contrast groups are heuristic. Near-miss, matched non-pattern, and
  typical-normal sets are interpretable, but not externally validated.

## Dataset Limitations

- Dataset provenance is UNKNOWN.
- Dataset licensing and ethics are UNKNOWN.
- Ground truth labels are UNKNOWN.
- The meaning of `label` in point-level files is UNKNOWN.
- The relationship between GeoJSON file counts and trajectory feature rows is
  unclear for fox and AIS.
- The hurricanes legacy outlier-score file has one fewer row than the current
  trajectory feature table.

## Paper Framing Risks

- Avoid claiming "Jumping Emerging Patterns" for the current Step 2 unless the
  JEP implementation is restored or the terminology is carefully revised.
  Current code mines FP-growth association rules over pseudo-label
  transactions.
- Avoid claiming formal statistical significance unless bootstrap, null-model,
  or another validation procedure is implemented and run.
- Avoid claiming domain validity without expert or literature-backed
  interpretation of the discovered patterns.
- Avoid claiming scalability without runtime measurements.
- Avoid claiming taxonomy optimality without a design rationale or ablation.

## Questions for the Researcher

1. What are the sources, licenses, and intended semantics of the fox, AIS, and
   hurricanes datasets?
2. What is the scientific meaning of the point-level `label` column, and should
   it be used or ignored?
3. Should the paper call the current rule-mining stage FP-growth association
   mining, or do you want to restore a strict/near-JEP definition?
4. What domain rationale should justify the pseudo-label thresholds and the
   Hybrid/Uncommon/Normal regions?
5. Should Hybrid behavior be a secondary analysis target, given that Hybrid is
   the largest pseudo-label class in retained pair observations?
6. What is the intended evaluation criterion for "better" than baselines:
   compactness, interpretability, stability, expert plausibility, coverage,
   runtime, or another measure?
7. Are depth-1-only taxonomy comparisons the final methodological choice, or
   should leaf-level analyses remain as an ablation?
8. Which generated output line should be considered canonical for the paper:
   only `taxonomy_v2`, or also legacy `step2_association_refined` and
   `step3_counterexamples`?
9. Do we need to keep generated outputs in Git, or should the repository move
   to scripts plus reproducibility instructions and ignore outputs?
10. Should we add tests before the paper-writing phase to lock down behavior?

