import concurrent.futures as cf
import hashlib,io,json,random,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parent
PREV=Path('/root/challenge-new-controls-20261122-a')
sys.path.insert(0,str(PREV))
import prepare as base
OLD=base.OLD
SEED=2026112202
def encode(rows):return ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode()
def sha(data):return hashlib.sha256(data).hexdigest()
def locate(name):
    assert name.startswith('coco/') and Path(name).name[:-4].isdigit()
    for p in [ROOT/'images'/Path(name).name,OLD/'images'/name,PREV/'images'/Path(name).name]:
        if p.is_file():return p
    return ROOT/'images'/Path(name).name
def verify_image(name):
    p=locate(name)
    data=p.read_bytes() if p.is_file() else base.fetch('https://s3.amazonaws.com/images.cocodataset.org/val2017/'+p.name,p)
    with Image.open(io.BytesIO(data)) as im:
        y=np.asarray(ImageOps.exif_transpose(im).convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
    basis=np.cos(np.pi*(np.arange(32)+.5)*np.arange(8)[:,None]/32)
    a=(basis@y@basis.T).ravel()[1:]
    return sha(data),int(''.join('1' if b else '0' for b in a>np.median(a)),2)
def main():
    (ROOT/'images').mkdir(exist_ok=True)
    olds=json.loads((OLD/'results/m0-20261102/image-audit.json').read_text())['files']
    prior={n:r['sha256'] for n,r in olds.items() if n.startswith('coco/')}
    prior.update({r['images'][0]:r['image_sha256'][0] for r in base.jsonlines(OLD/'runs/siglip-natural-full/predictions.jsonl')})
    prev=base.jsonlines(PREV/'manifest.private.jsonl')
    prior.update({r['img']:r['sha256'] for r in prev})
    assert len(prior)==820
    used=set(prior);bytehash=set(prior.values());phas=[]; skipped=[];allrows=[]
    with cf.ThreadPoolExecutor(8) as pool:
        for n,(d,h) in zip(sorted(prior),pool.map(verify_image,sorted(prior))):
            assert d==prior[n];phas.append(h)
        print('verified old images',len(used),flush=True)
        for cat,oid in base.OIDS.items():
            d=(PREV/(cat+'.json')).read_bytes()
            assert hashlib.sha1(f'blob {len(d)}\0'.encode()+d).hexdigest()==oid
            rows=sorted(json.loads(d),key=lambda r:str(r['id']))
            random.Random(f'{SEED}/{cat}/sample').shuffle(rows)
            source=iter(rows); selected=[]
            while len(selected)<30:
                batch=[];names=set()
                while len(batch)<min(10,30-len(selected)):
                    r=next(source)
                    name='coco/'+r['filename']
                    texts=[r['caption'],r['caption2'],r['negative_caption']]
                    if name in used or name in names:continue
                    if len(set(texts))!=3 or not all(isinstance(x,str) and x.strip() for x in texts):
                        skipped.append([name,'caption']);continue
                    batch.append((name,r['id'],texts));names.add(name)
                for (n,i,t),(d,h) in zip(batch,pool.map(verify_image,[b[0] for b in batch])):
                    if d in bytehash or any((h^p).bit_count()<=4 for p in phas):
                        used.add(n);skipped.append([n,'duplicate']);continue
                    used.add(n);bytehash.add(d);phas.append(h)
                    selected.append(dict(id=f'scpp/{cat}/{i}',cat=cat,img=n,sha256=d,phash=f'{h:016x}',texts=t))
            rng=random.Random(f'{SEED}/{cat}/wrong');rng.shuffle(selected);offsets=rng.sample(range(1,30),3)
            for i,r in enumerate(selected):r['wrong']=[selected[(i+k)%30]['img'] for k in offsets]
            allrows.extend(selected);print(cat,len(selected),flush=True)
    random.Random(f'{SEED}/presentation').shuffle(allrows)
    data=encode(allrows)
    p=ROOT/'base.private.jsonl'
    if p.exists(): assert p.read_bytes()==data
    else:p.write_bytes(data)
    stats=dict(n=150,prior=820,seed=SEED,sha256=sha(data),skips=skipped,source_commit=base.PIN,source_blobs=base.OIDS)
    (ROOT/'base.audit.json').write_text(json.dumps(stats,indent=2))
    print('cohort frozen',sha(data),flush=True)
if __name__=='__main__':main()
