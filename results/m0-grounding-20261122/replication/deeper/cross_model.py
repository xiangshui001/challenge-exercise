"""Cross-model agreement on matched persistent errors in the sealed 200-image cohort.

Run with Python and NumPy from any directory. No models or images are loaded.
"""
from pathlib import Path
import hashlib
import json
import re
from collections import Counter

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "controls"
ARMS = ("original", "wrong1", "wrong2", "wrong3", "text")
SEED = 2026112204
DRAWS = 30000


def read_model(name):
    path = DATA / f"full_{name}.jsonl"
    seal = json.loads((DATA / f"full_{name}.run.json").read_text(encoding="utf8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == seal["predictions_sha256"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    rows.sort(key=lambda row: row["id"])
    assert len(rows) == len({r["id"] for r in rows}) == seal["n"] == 200
    scores = np.asarray([[r["scores"][arm] for arm in ARMS] for r in rows])
    assert scores.shape == (200, 5, 3) and np.isfinite(scores).all()
    failure = scores[:, :, :2] <= scores[:, :, 2, None]
    original = failure[:, 0]
    persistent = original & (failure[:, 1:4].sum(axis=1) >= 2)
    masks = {"original": original, "persistent": persistent,
             "joint_text": persistent & failure[:, 4], "text": failure[:, 4]}
    identity = [(r["id"], r["cat"], r["sha256"], r["wrong"], r["text_sha256"]) for r in rows]
    return identity, masks


def groups_for(rows):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(tuple(row), []).append(i)
    return [np.asarray(g, dtype=int) for g in groups.values()]


def stratified_counts(first, second, groups, seed):
    rng = np.random.default_rng(seed)
    counts = np.zeros(DRAWS, dtype=int)
    # Shuffling the second model's entire row leaves both caption comparisons together.
    for start in range(0, DRAWS, 500):
        block = counts[start:start + 500]
        for g in groups:
            # Continuous random keys make independent row-wise random permutations.
            perms = np.argsort(rng.random((len(block), len(g))), axis=1)
            block += (first[g][None, :, :] & second[g][perms]).sum(axis=(1, 2))
    observed = int((first & second).sum())
    expected = sum(float(first[g].sum(axis=0) @ second[g].sum(axis=0)) / len(g) for g in groups)
    return {"observed": observed, "expected_exact": expected, "sampled_mean": float(counts.mean()),
            "central_95": np.quantile(counts, [.025, .975], method="inverted_cdf").tolist(),
            "greater_equal_plus_one_p": (int((counts >= observed).sum()) + 1) / (DRAWS + 1),
            "groups": len(groups), "singleton_groups": sum(len(g) == 1 for g in groups)}


def role_stats(mask):
    return {"reference_1": int(mask[:, 0].sum()), "reference_2": int(mask[:, 1].sum()),
            "reference_1_only": int((mask[:, 0] & ~mask[:, 1]).sum()),
            "reference_2_only": int((mask[:, 1] & ~mask[:, 0]).sum()),
            "both_references": int(mask.all(axis=1).sum())}


def overlap(first, second, categories):
    matched = first & second
    return {"common_pairs": int(matched.sum()), "common_images": int(matched.any(axis=1).sum()),
            "union_pairs": int((first | second).sum()), "reference_1": int(matched[:, 0].sum()),
            "reference_2": int(matched[:, 1].sum()),
            "by_category": {str(k): int(matched[categories == k].sum()) for k in np.unique(categories)}}


def token_set(text):
    return set(re.findall(r"\w+", text.casefold()))


def overlap_info(ids, raw):
    manifest = {r["id"]: r for r in map(json.loads, (DATA / "manifest.private.jsonl").read_text(encoding="utf8").splitlines())}
    diffs, word_margins = [], []
    for item, _, _, _, hashes in ids:
        texts = manifest[item]["texts"]
        assert [hashlib.sha256(t.encode()).hexdigest() for t in texts] == hashes
        t1, t2, foil = map(token_set, texts)
        j = lambda a,b: len(a & b) / len(a | b)
        diffs.append([j(t1, foil),j(t2, foil)])
        # When a sentence is scored by its mean similarity to the two others, the
        # score difference to the foil cancels their mutual similarity.
        word_margins.append([(j(t1,t2)-j(t2,foil))/2,(j(t1,t2)-j(t1,foil))/2])
    js = np.array(diffs)
    word = np.asarray(word_margins)
    fail = word <= 0
    wp = np.array([float(fail[:,r].mean()) for r in (0,1)])
    by_model = {}
    for name, masks in raw.items():
        outcomes = masks["text"]
        expected_by_role = sum(
            len(ids) * (wp[r]*outcomes[:,r].mean() + (1-wp[r])*(1-outcomes[:,r].mean()))
            for r in (0,1))
        both = (fail & outcomes)
        by_model[name] = {"agreement":int((fail==outcomes).sum()), "shared_failures":int(both.sum()),
                          "num_pairs":int(fail.size), "chance_agreement_preserving_roles":float(expected_by_role),
                          "agreement_by_role":(fail==outcomes).sum(axis=0).tolist(),
                          "shared_failures_by_role":both.sum(axis=0).tolist(),
                          "model_failures_by_role":outcomes.sum(axis=0).tolist()}
    return {"definition":"Lowercase alphanumeric/underscore tokens, set Jaccard distance complement; no punctuation or stemming.",
            "mean_foil_jaccard_by_ref":js.mean(axis=0).tolist(),
            "reference1_closer":int((js[:, 0] > js[:, 1]).sum()),
            "reference2_closer":int((js[:, 0] < js[:, 1]).sum()),
            "equal_distance":int((js[:, 0] == js[:, 1]).sum()),
            "parameter_free_jaccard_triplet_baseline":{
                "score":"Each candidate's mean Jaccard similarity to the other two; no image, learned model or reference labels for scoring.",
                "failures_or_ties_by_role":fail.sum(axis=0).tolist(), "ties_by_role":(word==0).sum(axis=0).tolist(),
                "both_win_items":int((~fail.any(axis=1)).sum()),"one_win_items":int((fail.sum(axis=1)==1).sum()),
                "neither_wins_items":int(fail.all(axis=1).sum()),"paired_with_text_scores":by_model},
            "discordant_failures_which_ref_closer_to_foil": {
                model: dict(Counter(
                    "more_similar" if js[i, j] > js[i, 1-j] else "less_similar" if js[i, j] < js[i, 1-j] else "tie"
                    for i,row in enumerate(masks["original"]) if row.sum() == 1
                    for j in [int(np.flatnonzero(row)[0])]
                )) for model, masks in raw.items()}}


def main():
    base, models = None, {}
    for name in ["siglip", "clip", "qwen"]:
        ids, models[name] = read_model(name)
        if base is not None:
            assert ids == base
        base = ids
    cats = np.array([row[1] for row in base])
    strata = groups_for(zip(cats))
    assert len(strata) == 5 and all(len(g) == 40 for g in strata)
    A, B = models["siglip"], models["clip"]
    summary = {"seed": SEED, "draws": DRAWS, "images":len(base), "reference_pairs":2*len(base),
               "models_reference_roles": {k:{n:role_stats(mask) for n,mask in v.items()} for k,v in models.items()},
               "lexical":overlap_info(base, models), "dual":{}, "all_three":{}}
    for k, arm in enumerate(["original", "persistent", "joint_text"]):
        summary["dual"][arm] = overlap(A[arm], B[arm], cats)
        summary["dual"][arm]["unconditional_by_category"] = stratified_counts(A[arm], B[arm], strata, SEED+k)
        if arm != "original":
            # Stronger diagnostic: completely fix which original pairs BOTH models miss.
            # CLIP's two-reference original-error mask stays on its original row
            # by restricting permutations to rows with the identical pattern in CLIP.
            conditioned = groups_for(zip(cats, B["original"][:, 0], B["original"][:, 1]))
            tally = stratified_counts(A[arm], B[arm], conditioned, SEED+10+k)
            conditioned_reverse = groups_for(zip(cats, A["original"][:, 0], A["original"][:, 1]))
            reverse = stratified_counts(B[arm], A[arm], conditioned_reverse, SEED+20+k)
            summary["dual"][arm]["conditional_clip_original_pattern"] = tally
            summary["dual"][arm]["conditional_siglip_original_pattern"] = reverse
        both = A[arm] & B[arm] & models["qwen"][arm]
        summary["all_three"][arm] = {"pairs":int(both.sum()), "images":int(both.any(axis=1).sum())}
    oo = A["original"] & B["original"]
    summary["dual"]["on_shared_original"] = {
        "pair_total":int(oo.sum()), "siglip_retained":int((oo & A["persistent"]).sum()),
        "clip_retained":int((oo & B["persistent"]).sum()),
        "both_retained":int((A["persistent"] & B["persistent"]).sum())}
    (ROOT/"cross_model.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
    for a in ("original", "persistent", "joint_text"):
        print(a,summary["dual"][a])
    print("roles",summary["models_reference_roles"])
    print("lexical",summary["lexical"])
    print("shared",summary["dual"]["on_shared_original"])
    print("three",summary["all_three"])

if __name__=="__main__":
    main()
