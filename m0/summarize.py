"""Recompute all reported metrics from public per-unit predictions."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from .metrics import sugar,wsd,classification,paired_bootstrap
from .prepare import save


def summarize(paths):
    groups=defaultdict(list);ids=set()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            r=json.loads(line)
            key=(r['benchmark'],r['model'],r['split'],r.get('arm','K0'))
            identity=(*key,r['sample_id'])
            if identity in ids:raise ValueError('Duplicate predictions; do not count smoke and final twice')
            ids.add(identity);groups[key].append(r)
    report={};pairs={}
    for key,rows in sorted(groups.items()):
        valid=[r for r in rows if r['status']=='ok']
        name='/'.join(key)
        if not valid:report[name]=dict(registered=len(rows),valid=0);continue
        if key[0]=='esnli':
            result=classification([r['gold'] for r in valid],[r['prediction'] for r in valid])
            result['accuracy_ci']=paired_bootstrap([int(r['gold']==r['prediction']) for r in valid])
            result['score_meaning']='original_gold_agreement_control' if key[3].endswith(('no_image','replace_image')) else 'official_gold_accuracy'
        else:
            calc=[sugar(r['scores'],r.get('text_scores')) if key[0]=='scpp' else wsd(r['scores'],r['gold']) for r in valid]
            result={m:paired_bootstrap([r[m] for r in calc]) for m in calc[0] if m!='prediction'}
            if key[0]=='scpp':
                swapped=[sugar(r['replacement_scores'])['itt'] for r in valid if 'replacement_scores' in r]
                if swapped:result['replacement_itt']=paired_bootstrap(swapped)
            result['truncated_texts']=sum(r['truncated'] for r in valid)
        result.update(registered=len(rows),valid=len(valid),
            unavailable=sum(r['status']=='unavailable' for r in rows),excluded=sum(r['status']=='excluded' for r in rows))
        report[name]=result
    # Complete-case paired comparisons explicitly aligned by immutable IDs.
    for split in ('exploration','confirmation'):
        for arm in ('K0','K1','Kshuffle','K2_no_image','K2_replace_image'):
            x=groups.get(('esnli','qwen',split,'K2'),[])
            y=groups.get(('esnli','qwen',split,arm),[])
            byid={r['sample_id']:r for r in y if r['status']=='ok'}
            both=[(r,byid[r['sample_id']]) for r in x if r['status']=='ok' and r['sample_id'] in byid]
            if not both:continue
            av=[int(a['prediction']==a['gold']) for a,b in both]
            bv=[int(b['prediction']==b['gold']) for a,b in both]
            pairs[split+'/K2-minus-'+arm]=dict(ci=paired_bootstrap(av,bv),
                repairs=[a['sample_id'] for a,b in both if a['prediction']==a['gold'] and b['prediction']!=b['gold']],
                regressions=[a['sample_id'] for a,b in both if a['prediction']!=a['gold'] and b['prediction']==b['gold']],
                unchanged_correct=sum(a and b for a,b in zip(av,bv)),unchanged_wrong=sum(not a and not b for a,b in zip(av,bv)))
        # K0 image dependence is separate from the K2 knowledge contrasts.
        real={r['sample_id']:r for r in groups.get(('esnli','qwen',split,'K0'),[]) if r['status']=='ok'}
        noimg={r['sample_id']:r for r in groups.get(('esnli','qwen',split,'K0_no_image'),[]) if r['status']=='ok'}
        common=sorted(real.keys() & noimg.keys())
        if common:
            pairs[split+'/K0-minus-K0_no_image']=dict(ci=paired_bootstrap(
                [real[i]['prediction']==real[i]['gold'] for i in common],
                [noimg[i]['prediction']==noimg[i]['gold'] for i in common]))
        for bench,metric in (('scpp','itt'),('vwsd','top1')):
            models={m:{r['sample_id']:r for r in groups.get((bench,m,split,'K0'),[]) if r['status']=='ok'} for m in ('siglip','clip')}
            common=sorted(models['siglip'].keys() & models['clip'].keys())
            if common:
                calc=lambda r:sugar(r['scores'])[metric] if bench=='scpp' else wsd(r['scores'],r['gold'])[metric]
                pairs[split+'/'+bench+'/siglip-minus-clip']=dict(ci=paired_bootstrap(
                    [calc(models['siglip'][i]) for i in common],[calc(models['clip'][i]) for i in common]))
    strata={};costs={}
    for key,rows in sorted(groups.items()):
        valid=[r for r in rows if r['status']=='ok']
        if key[0]=='scpp':
            strata['/'.join(key)]={c:dict(n=len(sub),itt=sum(sugar(r['scores'])['itt'] for r in sub)/len(sub))
                for c in sorted({r['category'] for r in valid}) if (sub:=[r for r in valid if r['category']==c])}
        if key[0]=='esnli' and valid:
            costs['/'.join(key)]={name:dict(mean=sum(r[name] for r in valid)/len(valid),min=min(r[name] for r in valid),max=max(r[name] for r in valid))
                for name in ('knowledge_tokens','knowledge_senses','knowledge_edges','input_tokens','output_tokens','inference_seconds')}
    return dict(groups=report,paired=pairs,descriptive_categories=strata,joint_tokens_and_time=costs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('predictions',nargs='+');p.add_argument('--out',required=True)
    a=p.parse_args();save(a.out,summarize(a.predictions))
