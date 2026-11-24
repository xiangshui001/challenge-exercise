"""Descriptive re-analysis of the sealed November 22 Qwen triplets; no access to human labels."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "controls"
ARMS = ["original", "wrong1", "wrong2", "wrong3", "text"]
SEED = 20261122


def interval(x):
    return np.quantile(x, [.025, .975], axis=0).tolist()


def summarize(x):
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                q25_q75=np.quantile(x, [.25, .75]).tolist())


def centered(array, groups):
    out = array.copy()
    for group in groups:
        out[group] -= array[group].mean(axis=0)
    return out


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def oof_predict(x, y, folds, groups):
    """Compare category-by-caption-slot means with those means plus one shared text slope."""
    cell = np.zeros((len(x), 2, 2 * len(groups)))
    for j, group in enumerate(groups):
        cell[group, 0, 2*j] = 1
        cell[group, 1, 2*j+1] = 1
    x1 = np.concatenate([cell, x[:, :, None]], axis=2)
    predictions = []
    slopes = []
    for design in (cell, x1):
        predicted = np.zeros_like(y)
        for f in range(5):
            train, test = folds != f, folds == f
            coef = np.linalg.lstsq(design[train].reshape(-1, design.shape[2]),
                                   y[train].ravel(), rcond=None)[0]
            predicted[test] = design[test] @ coef
            if design is x1:
                slopes.append(float(coef[-1]))
        predictions.append(predicted)
    return *predictions, slopes


def conditional_analysis(x, y, groups, folds, boot, perm):
    x_center, y_center = centered(x, groups), centered(y, groups)
    rho = corr(x_center, y_center)
    denom = np.sqrt(np.square(x_center).sum() * np.square(y_center).sum())
    rho_null = (x_center[perm] * y_center[None]).sum(axis=(1, 2)) / denom
    base, aug, slopes = oof_predict(x, y, folds, groups)
    squared_base = ((base-y)**2).mean(axis=1)
    squared_aug = ((aug-y)**2).mean(axis=1)
    change = squared_base - squared_aug
    return {
        "pearson_unadjusted": corr(x,y),
        "within_category_and_reference_slot_pearson":rho,
        "permutation_null_ci95_for_adjusted_r":interval(rho_null),
        "permutation_p_two_sided":float((1+(np.abs(rho_null)>=abs(rho)).sum())/(1+len(perm))),
        "held_out_baseline_rmse":float(np.sqrt(squared_base.mean())),
        "held_out_with_text_rmse":float(np.sqrt(squared_aug.mean())),
        "held_out_relative_mse_reduction":float(1-squared_aug.sum()/squared_base.sum()),
        "held_out_mse_reduction":float(change.mean()),
        "conditional_ci95_mse_reduction_fixed_predictions":interval(change[boot].mean(axis=1)),
        "fitted_text_slopes_training_folds":slopes,
    }


def paired_change(before, after, boot):
    gained=(~before)&after
    lost=before&(~after)
    image_delta=(after.astype(float)-before.astype(float)).reshape(len(before),-1).mean(axis=1)
    out=dict(before=int(before.sum()), after=int(after.sum()), repaired=int(gained.sum()),
             regressed=int(lost.sum()), n=int(before.size),
             difference=float(image_delta.mean()), stratified_image_bootstrap_ci95=interval(image_delta[boot].mean(axis=1)))
    if before.ndim==1:
        n=int(gained.sum()+lost.sum())
        least=int(min(gained.sum(),lost.sum()))
        out["exact_two_sided_sign_p"]=min(1.,2*sum(math.comb(n,i) for i in range(least+1))/2**n)
    return out


def main():
    p=SOURCE/'full_qwen.jsonl'
    run=json.loads((SOURCE/'full_qwen.run.json').read_text(encoding="utf8"))
    checksum=hashlib.sha256(p.read_bytes()).hexdigest()
    assert checksum==run['predictions_sha256']
    rows=[json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]
    assert len(rows)==len({r["id"] for r in rows})==200
    categories=sorted({r['cat'] for r in rows})
    groups=[np.array([i for i,r in enumerate(rows) if r["cat"]==c]) for c in categories]
    assert all(len(g)==40 for g in groups)
    scores=np.array([[r['scores'][arm] for arm in ARMS] for r in rows],dtype=float)
    assert scores.shape==(200,5,3) and np.isfinite(scores).all()
    margins=scores[:,:,:2]-scores[:,:,2,None]
    orig=margins[:,0,:]
    wrong=margins[:,1:4,:]
    text=margins[:,4,:]
    wrong_mean=wrong.mean(axis=1)
    err=orig<=0; repeats=err&((wrong<=0).sum(axis=1)>=2); joint=repeats&(text<=0)
    assert [int(m.sum()) for m in (err,repeats,joint)]==[31,13,8]
    f=np.empty(200,dtype=int)
    for group in groups:
        ordered=sorted(group,key=lambda i:hashlib.sha256(("oof-bias-v1/"+rows[i]['id']).encode()).digest())
        f[ordered]=np.arange(40)%5
    rng=np.random.default_rng(SEED)
    boot=np.concatenate([rng.choice(g,(5000,len(g)),replace=True) for g in groups],axis=1)
    perm=np.empty((5000,200),dtype=int)
    for group in groups:
        perm[:,group]=group[np.argsort(rng.random((5000,len(group))),axis=1)]
    out={"source_sha256":checksum,"N_images":200,"seed":SEED,
         "description":"Each row contributes two reference-minus-foil margins. Positive logit difference means Yes is preferred over No, conditional on comparing those two tokens. OOF folds stratify by five official edit types; image is the unit. Per-task tests are exploratory. For image permutations all caption slots move together within category; replacement mappings are fixed. Percentile intervals are descriptive, with resampling by image within edit type. OOF loss intervals condition on the fitted folds.",
         "arms":{},"conditional_groups":{},"linear_text_signal":{},"fixed_calibration":{},"wrong_margin_repeatability":{}}
    for i,arm in enumerate(ARMS):
        z=scores[:,i]
        out["arms"][arm]=dict(yes=int((z>0).sum()),no=int((z<0).sum()),ties=int((z==0).sum()),
                              yes_by_slot=(z>0).sum(0).tolist(),median_by_slot=np.median(z,axis=0).tolist(),
                              median_margin=float(np.median(margins[:,i])),median_abs_margin=float(np.median(abs(margins[:,i]))),
                              margin=summarize(margins[:,i]))
    for name,mask in (("all",np.ones_like(err)),("original_errors",err),("persistent",repeats),
                      ("persistent_and_text",joint),("correct_original",~err)):
        o=scores[:,0,2,None] * np.ones((1,2)); t=scores[:,4,2,None] * np.ones((1,2))
        out["conditional_groups"][name]=dict(pairs=int(mask.sum()),images=int(mask.any(1).sum()),
            original_margin=summarize(orig[mask]),wrongmean_margin=summarize(wrong_mean[mask]),
            text_margin=summarize(text[mask]),positive_foil_original=int((o[mask]>0).sum()),
            positive_foil_text=int((t[mask]>0).sum()))
    for name,y in (("text_predict_original",orig),("text_predict_mean_wrong",wrong_mean),
                   ("text_predict_wrong1",wrong[:,0]),("text_predict_wrong2",wrong[:,1]),("text_predict_wrong3",wrong[:,2])):
        out["linear_text_signal"][name]=conditional_analysis(text,y,groups,f,boot,perm)
    for i in range(3):
        for j in range(i+1,3):
            out["wrong_margin_repeatability"][f"wrong{i+1}_wrong{j+1}"]=corr(centered(wrong[:,i],groups),centered(wrong[:,j],groups))
    for name,ref in (("noimage",text),("mean_three_wrong",wrong_mean)):
        corrected=orig-ref
        out['fixed_calibration'][name]=dict(
            pairs=paired_change(orig>0,corrected>0,boot),
            images_both_win=paired_change((orig>0).all(1),(corrected>0).all(1),boot),
            by_category={cat:{"raw":int((orig[g]>0).all(1).sum()),"corrected":int((corrected[g]>0).all(1).sum())} for cat,g in zip(categories,groups)})
        for transition,mask in (("repaired_pairs",(orig<=0)&(corrected>0)),("regressed_pairs",(orig>0)&(corrected<=0))):
            out["fixed_calibration"][name][transition]={
                key:summarize(value[mask]) for key,value in (("raw_margin",orig),("subtracted_margin",ref),("corrected_margin",corrected))
            }
    target=HERE/'offsets.json'
    target.write_bytes(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False).encode("utf8"))
    print(json.dumps({k:out[k] for k in ("linear_text_signal","fixed_calibration","wrong_margin_repeatability")},indent=1))

if __name__=="__main__":
    main()
