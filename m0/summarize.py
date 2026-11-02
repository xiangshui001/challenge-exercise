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
        else:
            calc=[sugar(r['scores'],r.get('text_scores')) if key[0]=='scpp' else wsd(r['scores'],r['gold']) for r in valid]
            result={m:paired_bootstrap([r[m] for r in calc]) for m in calc[0] if m!='prediction'}
            if key[0]=='scpp':
                swapped=[sugar(r['replacement_scores'])['itt'] for r in valid if 'replacement_scores' in r]
                if swapped:result['replacement_itt']=paired_bootstrap(swapped)
            result['truncated_texts']=sum(r['truncated'] for r in valid)
        result.update(registered=len(rows),valid=len(valid),unavailable=len(rows)-len(valid))
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
    return dict(groups=report,paired=pairs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('predictions',nargs='+');p.add_argument('--out',required=True)
    a=p.parse_args();save(a.out,summarize(a.predictions))
