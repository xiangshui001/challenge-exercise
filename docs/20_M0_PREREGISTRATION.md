# M0 first run: frozen protocol, 2026-11-02

Registered before model inference. Base main: b50f74cd96bd2f6c16e80626b1130cd95e629621. No training. Independent branch; PR #3/#5 untouched. Public release contains IDs/hashes/scores, not images or unrestricted redistribution of annotations.

## Sampling and scope

Seed 2026. 120 SugarCrepe++ units (24 per five published categories, 14 exploration + 10 confirmation), 120 e-ViL e-SNLI-VE official test units (40 per class, 24 exploration + 16 confirmation), 40 English Visual-WSD official test units (24 + 16). Winoground omitted to keep the first run bounded. Choose distinct images across units; VWSD also distinct target words and disjoint candidate sets. Audit byte duplicates after acquisition. No substitution for download failures. These are small internal exploration/confirmation subsets, not full official benchmark results. Pretraining contamination unknown.

Sources: scpp c083bbf039a5ca215acaf25e7ee736d2ea6f3864; e-ViL 94f17f5c189573c9d1087f82eb27db37690fc810 (author-directed successor of deprecated e-SNLI-VE); official VWSD test v1.1 queries and resized images. Source hashes and IDs saved by `python -m m0.prepare` before scoring.

## Models and costs

Frozen SigLIP-base-patch16-224 at 7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed; independent CLIP ViT-B/32 at 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268. Raw unprompted captions; native processor; FP32 cosine scores, no head/calibration. Qwen2.5-VL-3B-Instruct at 66285546d2b821cf421d4f5eb2576359d3770cd3 for three-way visual entailment, BF16 greedy generation, maximum 16 new tokens and 262144 image pixels. One fixed prompt, no outcome-dependent revision. These are accessible frozen baselines, not a claim of frontier model coverage.

Only assigned physical GPUs 4/5, resolve UUID before CUDA startup; no multi-GPU auto placement. First 20 exploration units per applicable model smoke test. Smoke is an implementation check, excluded from the final aggregate; its repeated computation is counted in run costs. No tuning from smoke scores. Budget: at most 2 dual encoders + 1 3B joint model; <12 GB new weights, <8 GB source images, <1500 joint generations, zero paid API calls. Stop and report unavailable sources rather than invent results.

## Metrics, controls, and knowledge

SugarCrepe++ strict ITT: both positive cosine scores exceed negative; report each positive accuracy, XOR rank instability, both-wrong, official TOT (positive-positive exceeds BOTH positive-negative text cosines). Equal scores count as failures. Deterministic cyclic image replacement within split; never call TOT equivalent to no-image three-way reasoning. VWSD top-1 (first candidate wins exact ties, ties counted), MRR supplementary. e-SNLI-VE: three-class accuracy, macro F1, confusion matrix, neutral-to-entailment false support; invalid outputs count incorrect in denominator and get their own column.

Paired bootstrap, seed 2026, 2000 replicates, image/unit groups, separate datasets and splits. No averaged cross-benchmark score. Report all repairs/regressions/unchanged. Small confirmation sets give wide intervals. Exploratory multiple categories are descriptive, not a family of significance claims.

Knowledge pilot on all 120 preselected e-SNLI-VE units, with exploration/confirmation reported separately, if images/model are available. This predeclared all-unit pilot avoids selecting errors from confirmation. WordNet 3.0 via NLTK: retrieve from hypothesis only, no gold/premise captions/explanations; deterministic first two distinct alphabetic tokens with noun senses, at most first two senses each, at most two immediate hypernyms sorted by ID. This limited unsupervised sense retrieval is deliberately NOT an oracle. Include sense IDs, lemma sets, definitions and directed hypernym edges in both K1 prose and K2 triples. Same facts and ordering. Report actual token counts; equal information does not guarantee equal tokens.

Arms: K0 no knowledge, K1 prose, K2 typed triples, Kshuffle identical K2 facts with hypernym direction reversed (explicit corrupted-direction control, not random truths), K0 no image, K2 no image, K2 cyclic replacement image. One call per arm/unit; max tokens and decode fixed. Lexical facts do not establish visual truth. K2 must exceed K1 and corrupted control on held-out units and depend on image evidence before claiming independent value. If no valid edges, record empty coverage. No human oracle input unless separately supplied by an actual reviewer.

## Review and decision

Freeze official gold. Prepare blinded review queue and inspect image/caption cases as an AI-assisted single review, clearly distinguished from human review. No fabricated human agreement; human review remains pending unless performed by people. Record annotation ambiguities without editing official labels. Review disagreements and a deterministic sample of successes to avoid treating every model error as clean evidence.

H1: ranking failures establish benchmark-specific instability, semantic explanation requires review. H2: swapped relation failures suggest compositional difficulty, not proof of all directional entailment defects. H3 needs valid visual three-way results, especially neutral cases; cosine scores cannot answer H3. H4 remains unsupported if K2 fails to beat equal-fact prose or if noise/confounds dominate. Do not train a new module merely because one small point estimate improves.

## Pre-inference amendment: image-byte audit

Before the first model inference, the downloaded VWSD bytes exposed two duplicate candidate pairs despite different filenames: units 58/331 (cross split), and 48/167 (both exploration). Quarantine all four units from primary scoring, retain their original IDs and candidate lists, and do not replace them. A separate `exclusions.json` records this deterministic data-only exclusion, included in run provenance. Effective VWSD is 21 exploration + 15 confirmation. Exact duplicates are checked; perceptual near-duplicates and pretraining overlap remain unknown. The smoke-accounting wording above was clarified before inference to count repeated execution cost and avoid counting smoke twice in metrics.
