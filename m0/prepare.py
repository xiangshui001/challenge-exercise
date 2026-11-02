"""Freeze official IDs BEFORE inference; never resample after download/model failure."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8',newline='\n')


def sample_strata(strata, n, exploration, rng):
    """Unique images globally; shuffled round-robin strata, then per-stratum split."""
    chosen, used = [], set()
    pools = {k: sorted(v, key=lambda r:r['sample_id']) for k,v in strata.items()}
    for pool in pools.values():
        rng.shuffle(pool)
    for i in range(n):
        for key, pool in pools.items():
            while pool:
                row = pool.pop()
                if not used.intersection(row['images']):
                    break
            else:
                raise ValueError(f'Not enough distinct image groups: {key}')
            used.update(row['images'])
            row['split'] = 'exploration' if i < exploration else 'confirmation'
            chosen.append(row)
    return chosen


def prepare(scpp, esnli, vwsd, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(2026)
    sources, normalized_sources, strata = {}, {}, {}
    for p in sorted(Path(scpp).glob('*.json')):
        sources[p.name] = sha(p)
        normalized_sources[p.name]=hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
        strata[p.stem] = [dict(sample_id=f'scpp/{p.stem}/{r["id"]}', benchmark='scpp',
            category=p.stem, images=[f'coco/{r["filename"]}'],
            texts=[r['caption'],r['caption2'],r['negative_caption']], gold=[0,1],
            source_id=str(r['id']), official_split='published_evaluation', review='unknown')
            for r in json.loads(p.read_text(encoding='utf-8'))]
    rows = sample_strata(strata,24,14,rng)
    sources['esnlive_test.csv'] = sha(esnli)
    normalized_sources['esnlive_test.csv']=hashlib.sha256(Path(esnli).read_text(encoding='utf-8').encode()).hexdigest()
    strata = {k:[] for k in ('contradiction','entailment','neutral')}
    with Path(esnli).open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            strata[r['gold_label']].append(dict(sample_id='esnli/'+r['pairID'], benchmark='esnli',
                category=r['gold_label'], images=['flickr/'+r['Flickr30kID']], texts=[r['hypothesis']],
                gold=r['gold_label'], source_id=r['pairID'], official_split='test', review='unknown'))
    rows += sample_strata(strata,40,24,rng)
    qp, gp = [Path(vwsd)/x for x in ('en.test.data.v1.1.txt','en.test.gold.v1.1.txt')]
    sources[qp.name],sources[gp.name] = sha(qp),sha(gp)
    for p in (qp,gp):normalized_sources[p.name]=hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
    queries,golds = qp.read_text().splitlines(),gp.read_text().splitlines()
    if len(queries)!=len(golds): raise ValueError('VWSD length mismatch')
    pool=[]
    for i,(q,g) in enumerate(zip(queries,golds)):
        target,phrase,*images=q.split('\t')
        if len(images)!=10 or g not in images: raise ValueError('VWSD candidates/gold')
        pool.append(dict(sample_id=f'vwsd/en/{i+1}',benchmark='vwsd',category='en',target=target,
            images=['vwsd/'+x for x in images],texts=[phrase],gold=images.index(g),
            source_id=str(i+1),official_split='test',review='unknown'))
    # Reject reuse of target or candidate images across selected units, before scores exist.
    rng.shuffle(pool)
    chosen,used,words=[],set(),set()
    for r in pool:
        if used.intersection(r['images']) or r['target'] in words: continue
        r['split']='exploration' if len(chosen)<24 else 'confirmation'
        chosen.append(r); used.update(r['images']); words.add(r['target'])
        if len(chosen)==40: break
    if len(chosen)!=40: raise ValueError('Insufficient VWSD disjoint units')
    rows += chosen
    (out/'manifest.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8',newline='\n')
    public=[]
    for r in rows:
        p={k:v for k,v in r.items() if k not in ('texts','target')}
        p['text_sha256']=[hashlib.sha256(t.encode()).hexdigest() for t in r['texts']]
        public.append(p)
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in public),encoding='utf-8',newline='\n')
    save(out/'freeze.json',dict(seed=2026,manifest_sha256=sha(out/'manifest.private.jsonl'),
        public_manifest_sha256=sha(out/'manifest.jsonl'),source_sha256=sources,
        normalized_source_sha256=normalized_sources,
        units=len(rows),note='Frozen before image acquisition and model inference; missing files stay missing.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('scpp','esnli','vwsd','out'): p.add_argument('--'+arg,required=True)
    a=p.parse_args(); prepare(a.scpp,a.esnli,a.vwsd,a.out)
