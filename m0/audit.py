"""Audit acquired bytes without removing or replacing registered samples."""
import argparse
import json
from pathlib import Path
from .prepare import sha,save


def audit(manifest,images):
    from PIL import Image
    rows=[json.loads(x) for x in Path(manifest).read_text().splitlines()]
    seen={}; missing=[]; files={};counts={}
    for r in rows:
        counts.setdefault(r['benchmark'],{}).setdefault(r['split'],0)
        counts[r['benchmark']][r['split']]+=1
        for name in r['images']:
            p=Path(images)/name
            if not p.exists():missing.append(dict(sample_id=r['sample_id'],image=name));continue
            h=sha(p)
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:shape=list(im.size)
            files[name]=dict(sha256=h,bytes=p.stat().st_size,size=shape)
            seen.setdefault(h,[]).append((r['sample_id'],r['split'],name))
    duplicates={h:v for h,v in seen.items() if len({x[0] for x in v})>1}
    ep=Path(manifest).with_name('exclusions.json')
    excluded=json.loads(ep.read_text()) if ep.exists() else {}
    active_duplicates={h:v for h,v in duplicates.items() if len({x[0] for x in v if x[0] not in excluded})>1}
    return dict(manifest_sha256=sha(manifest),counts=counts,files=files,missing=missing,
        byte_duplicate_units=duplicates,cross_split_duplicate_groups=sum(len({x[1] for x in v})>1 for v in duplicates.values()),
        exclusions=excluded,active_byte_duplicate_groups=len(active_duplicates),
        pretraining_contamination='unknown',perceptual_duplicates='not audited',
        status=('passed_with_quarantine' if excluded else 'passed') if not missing and not active_duplicates else 'needs_review')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('manifest','images','out'):p.add_argument('--'+n,required=True)
    a=p.parse_args();result=audit(a.manifest,a.images);save(a.out,result)
    print({k:v for k,v in result.items() if k not in ('files','missing','byte_duplicate_units')})
