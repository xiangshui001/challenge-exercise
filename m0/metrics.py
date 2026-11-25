"""Task-specific metrics with strict ties and paired unit bootstrap."""
import numpy as np


def sugar(scores, text_scores=None):
    p1,p2,n=map(float,scores)
    if not np.isfinite([p1,p2,n]).all(): raise ValueError('Nonfinite scores')
    a,b=p1>n,p2>n
    result=dict(itt=int(a and b),p1=int(a),p2=int(b),xor=int(a!=b),
                both_wrong=int(not a and not b),margin1=p1-n,margin2=p2-n)
    if text_scores is not None:
        pp,pn1,pn2=map(float,text_scores)
        result['tot']=int(pp>pn1 and pp>pn2)
    return result


def wsd(scores,gold):
    if len(scores)!=10 or not 0<=gold<10 or not np.isfinite(scores).all():
        raise ValueError('Invalid ten-candidate WSD input')
    order=sorted(range(10),key=lambda i:(-scores[i],i))
    return dict(top1=int(order[0]==gold),mrr=1/(order.index(gold)+1),
                top_tie=int(sum(x==max(scores) for x in scores)>1),prediction=order[0])


LABELS=('contradiction','entailment','neutral')


def classification(gold,pred):
    if not gold or len(gold)!=len(pred) or any(g not in LABELS for g in gold):
        raise ValueError('Invalid classification data')
    confusion={g:{p:0 for p in (*LABELS,'invalid')} for g in LABELS}
    for g,p in zip(gold,pred): confusion[g][p if p in LABELS else 'invalid']+=1
    f1=[]
    for label in LABELS:
        tp=confusion[label][label]
        fp=sum(confusion[g][label] for g in LABELS if g!=label)
        fn=sum(confusion[label].values())-tp
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    neutral=sum(confusion['neutral'].values())
    return dict(n=len(gold),accuracy=sum(g==p for g,p in zip(gold,pred))/len(gold),
        macro_f1=float(np.mean(f1)),confusion=confusion,
        neutral_false_support=confusion['neutral']['entailment']/neutral if neutral else None,
        invalid=sum(p not in LABELS for p in pred))


def paired_bootstrap(values,reference=None,seed=2026,repeats=2000):
    """One row per disjoint original unit; align by ID before calling."""
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all(): raise ValueError('Invalid bootstrap values')
    if reference is not None:
        y=np.asarray(reference,dtype=float)
        if y.shape!=x.shape: raise ValueError('Unpaired sizes')
        x=x-y
    rng=np.random.default_rng(seed)
    samples=x[rng.integers(0,len(x),(repeats,len(x)))].mean(axis=1)
    low,high=np.quantile(samples,[.025,.975])
    return dict(estimate=float(x.mean()),low=float(low),high=float(high),n=len(x),
                seed=seed,repeats=repeats,paired=reference is not None)
