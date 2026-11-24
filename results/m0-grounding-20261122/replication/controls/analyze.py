"""Recompute result summaries from three sealed per-item prediction files (Python 3, numpy)."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ARMS=['original','wrong1','wrong2','wrong3','text']
MODELS=['qwen','siglip','clip']
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ratio(a,b):return float(a/b) if b else None
def assess(model,root):
    run=json.loads((root/f'full_{model}.run.json').read_text(encoding='utf8'))
    path=root/f'full_{model}.jsonl'
    assert digest(path)==run['predictions_sha256'] and run['n']==200
    rows=[json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]
    assert len(rows)==len({r['id'] for r in rows})==200
    categories=np.array([r['cat'] for r in rows])
    groups={str(c):np.flatnonzero(categories==c) for c in sorted(set(categories))}
    assert len(groups)==5 and all(len(idx)==40 for idx in groups.values())
    scores=np.array([[r['scores'][a] for a in ARMS] for r in rows],dtype=float)
    assert scores.shape==(200,5,3) and np.isfinite(scores).all()
    margin=scores[:,:,:2]-scores[:,:,2,None]
    fail=margin<=0
    orig=fail[:,0,:]; wrong_n=fail[:,1:4,:].sum(axis=1); text=fail[:,4,:]
    persistent=orig & (wrong_n>=2)
    all3=orig & (wrong_n==3)
    joint=persistent & text
    def c(x):return int(x.sum())
    def details(idx):
        o,p,j,t=orig[idx],persistent[idx],joint[idx],text[idx]
        return dict(n=len(idx),orig_pairs=c(o),persistent_pairs=c(p),text_overlap=c(o&t),joint_pairs=c(j),
                    orig_items=c(o.any(axis=1)),persistent_items=c(p.any(axis=1)),joint_items=c(j.any(axis=1)),
                    orig_xor_items=c(o.sum(axis=1)==1),both_failed_items=c(o.all(axis=1)),both_won_items=c(~o.any(axis=1)))
    summ={'model':model,**details(np.arange(200)),'original_ties':c(margin[:,0,:]==0),
          'wrong_all3_pairs':c(all3),'retention_each':[c(orig & fail[:,a,:]) for a in range(1,4)],
          'orig_negative_strictly_wins_both':c((margin[:,0,:]<0).all(axis=1)),
          'categories':{k:details(I) for k,I in groups.items()},'arms':{}}
    if model=='qwen':
        repeat=persistent[:,None,:] & fail[:,1:4,:]
        foil=scores[:,1:4,2,None]
        summ['absolute_yes_no_diagnostic']={'repeated_pair_by_wrong_image_comparisons':c(repeat),
              'both_prefer_no':c(repeat & (foil<0)),
              'foil_prefers_yes_in_text_on_joint_pairs':c(joint & (scores[:,4,2,None]>0))}
    for a,arm in enumerate(ARMS):
        m=fail[:,a,:].sum(axis=1)
        summ['arms'][arm]=dict(both_win=c(m==0),one_win=c(m==1),neither_wins=c(m==2),
            pair_failures=c(fail[:,a,:]),ties=c(margin[:,a,:]==0))
    rng=np.random.default_rng(20261122)
    metrics=[];null=[]
    for b in range(4000):
        index=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups.values()])
        denom=orig[index].sum()
        if denom:
            metrics.append([persistent[index].sum()/denom,joint[index].sum()/denom])
        perm=np.arange(200)
        for g in groups.values():perm[g]=rng.permutation(g)
        null.append([np.sum(orig & (wrong_n[perm]>=2)),np.sum(orig & (wrong_n[perm]>=2) & text[perm])])
    summ['bootstrap_fraction_ci95']=np.quantile(metrics,[.025,.975],axis=0).T.tolist() if metrics else []
    summ['permuted_pair_counts']={'mean':np.mean(null,axis=0).tolist(),'central95':np.quantile(null,[.025,.975],axis=0).T.tolist()}
    audit=[]
    for i,r in enumerate(rows):
        for j in (0,1):
            audit.append(dict(model=model,id=r['id'],category=r['cat'],reference=j+1,
                orig_error=int(orig[i,j]),wrong1_error=int(fail[i,1,j]),wrong2_error=int(fail[i,2,j]),wrong3_error=int(fail[i,3,j]),
                text_error=int(text[i,j]),same_error_after_swap=int(persistent[i,j]),persists_without_image=int(joint[i,j]),
                orig_tie=int(margin[i,0,j]==0),orig_margin=float(margin[i,0,j])))
    return rows,summ,audit
def main():
    combined=[];stats={}; prior=None
    for m in MODELS:
        rows,stats[m],audit=assess(m,HERE);combined+=audit
        identity=[(r['id'],r['sha256'],r['wrong'],r['text_sha256']) for r in rows]
        if prior is not None:assert identity==prior
        prior=identity
    (HERE/'summary.json').write_bytes(json.dumps(stats,ensure_ascii=False,indent=2).encode('utf8'))
    with (HERE/'pair_errors.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
    for m,s in stats.items():print(m,{k:s[k] for k in ('orig_pairs','persistent_pairs','text_overlap','joint_pairs','orig_items','joint_items','retention_each','original_ties')})
if __name__=='__main__':main()
