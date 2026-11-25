# M0 sources, rights, and existing assets

## Sources verified for this run

| Resource | Selected release and identity | Access / rights / scope |
|---|---|---|
| [SugarCrepe++](https://github.com/Sri-Harsha/scpp) | c083bbf039a5ca215acaf25e7ee736d2ea6f3864, five `data/*.json` files, 4,757 rows | Dataset README/HF card says CC BY 4.0; code LICENSE is MIT. Image rights remain separate. Official scripts use COCO val2017. Public results contain IDs/hashes, not images. |
| [SNLI-VE](https://github.com/necla-ml/SNLI-VE) | Source/task definitions and official author-linked Flickr30K archive audited | Its cosine similarity is not an entailment probability. Original paper notes label noise; preserve official and reviewer labels separately. |
| [e-SNLI-VE successor e-ViL](https://github.com/maximek3/e-ViL) | 94f17f5c189573c9d1087f82eb27db37690fc810, `data/esnlive_test.csv` | The older virginie-do repository explicitly says deprecated and redirects here. No blanket repository license observed for redistributing all annotations; do not republish CSV/explanations/images. IDs and derived statistics only. |
| [Visual-WSD](https://raganato.github.io/vwsd/) | Official English test data v1.1 and official resized test images | CC BY-NC 4.0; maintain research use and do not reverse-search images. 10 candidates, preserve their order. Google Drive IDs: query `10vDZsY0EhzvFFR8IF-3P_2ApOF0GIMML`, images `15ed8TXY9Pzk68_SCooFm7AfkeFtCd16Q`. |
| [WordNet](https://wordnet.princeton.edu/license-and-commercial-use) | WordNet 3.0 via NLTK `wordnet.zip`; archive hash recorded in each joint-model run | Princeton license permits use with required notice. No hypothesis gold or image captions used for sense retrieval. Candidate senses are not validated disambiguations. |

All public manifests contain complete source IDs, text hashes, original dataset split, internal split, official gold and review status. The private manifest retains texts locally. Source byte hashes reflect checkout line endings; normalized source hashes and canonical LF manifests enable cross-platform comparisons. Windows-to-Linux serialization was normalized before inference; all 280 parsed sample objects were checked identical. Earlier Git commits preserve the original registration.

## Data checks and exclusions

Sample selection is fixed before downloads. No resampling for inconvenient examples. COCO has 120 distinct image names. e-SNLI-VE selects 120 distinct Flickr IDs, class-balanced. VWSD initially selects 40 distinct targets with disjoint image filenames. Byte audit then finds two pairs of identical candidate images under different names: `en/58` vs `en/331`, and `en/48` vs `en/167`. All four units are quarantined before any model inference. Counts become 21 exploration / 15 confirmation; all original registrations stay visible. Exact-byte checking cannot rule out perceptual duplicates or training contamination.

Human labels remain `unknown`. AI visual inspection is not a human review and must not be reported as inter-annotator agreement. Official metrics remain unchanged by review comments.

## Historical assets: independently checked, not M0 scores

The user-authorized independent local CLI checked the old server through its already authenticated Xshell session on 2026-11-02. It was launched with `gpt-6.1-sol`, reasoning `xhigh`, service tier `fast`, using its existing separate ChatGPT login. No account switch, training, or GPU inference occurred there. Host addresses, credentials and raw terminal transcripts remain private.

Verified historical assets: SigLIP snapshot `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`, seven dereferenced files totaling 815,871,927 bytes; 31,783 JPEGs; Karpathy annotations; S0 cached features and per-query ranks. SigLIP safetensors SHA256 `2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8`. Saved T2I hit counts were 3,833 / 4,708 / 4,859 at k=1/5/10 over 5,070 queries, consistent with the old validation report. This verifies stored ranking summaries, not a new forward pass or recomputed ranking from embeddings.

Old repo-baseline HEAD was `65e1d387c989cd0189321f1855a29a782c9ecfe0`; repo/repo-audit HEAD `9f4d08085aebd4d1aed8bf35eeac3e0f9218fc67`. Tracked files were clean; ignored artifacts were preserved. GitHub PR #3 remains open and PR #5 draft/open, with #5 targeting the #3 branch. Current GitHub mergeability metadata may differ from historical conflict notes; no merge was attempted.

The exact 120 selected Flickr30K JPEGs were subsequently copied through the authenticated session to a separate local transfer drive: 16,043,886 bytes, all remote/local sizes and SHA256 values matched. Only those files were forwarded to the new evaluation workspace. Existing old weights/caches/results were not altered. New input hashes are public in `image-audit.json`; private server paths and transfer transcripts are not published. The CLI launch parameters were observed; the provider's actual response service tier was not independently verified.
