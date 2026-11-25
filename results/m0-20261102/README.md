# Frozen M0 evidence, 2026-11-02

Read [the research judgment](../../docs/M0_REPORT.md), [preregistration](../../docs/20_M0_PREREGISTRATION.md), and [reproduction guide](../../docs/22_M0_REPRODUCE.md).

- `manifest.jsonl`, `freeze.json`, `exclusions.json`: all 280 registered IDs and split/input identities; 4 VWSD units quarantined before inference.
- `image-audit.json`: 640 acquired file identities, decode checks, and duplicate audit. No image pixels are redistributed.
- `*-predictions.jsonl`: raw full-run scores or literal short model outputs. SigLIP/CLIP each have 160 records (156 valid); Qwen has 840 records across 7 arms. Official gold is retained, including disputed cases.
- `*-run.json`, `runtime-environment.json`: actual model hashes, code hashes, processors, dependencies, timing and assigned GPU identity. `*-smoke-run.json` records separate implementation-check cost; smoke is not aggregated into scores.
- `summary.json`: recomputed metrics, bootstrap intervals, ID-aligned repairs/regressions, descriptive categories and token/time accounting. Controls with absent/replaced images measure agreement with **original** labels, not newly annotated control truth.
- `knowledge-audit.json`: all 840 prompts reconstructed from private facts; K2 and corrupted prompts differ for all 120 units, while their predicted labels are identical.
- `ai-review.jsonl`: 28 initially sampled + 4 explicitly post-hoc visual inspections by one AI reviewer. Human review remains pending; not a replacement gold dataset or noise-rate estimate.
- `verification.json`, `tests.txt`, `validation.json`: actual QA evidence and public-file fingerprints.

The current inference implementations are represented by hashes in each run record; analysis/reporting helpers were completed after inference. No score-dependent model/prompt/retrieval change was made. Private manifests, dictionary instances, original annotations, weights, transfer transcripts and credentials are not published. Dataset rights and fixed source revisions are described in [the source audit](../../docs/21_M0_DATA_AND_ASSETS.md).
