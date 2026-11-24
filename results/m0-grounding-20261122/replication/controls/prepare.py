from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime,timezone
from io import BytesIO
import hashlib,itertools,json,random,time,urllib.request
import numpy as np
from PIL import Image,ImageOps

OLD=Path('/root/m0-followup-20261119')
ROOT=Path(__file__).resolve().parent
PIN='c083bbf039a5ca215acaf25e7ee736d2ea6f3864'
OIDS=dict(swap_obj='8e9c3f66a2c2d3369146dec5324b1926b2fc1d40',swap_att='61ea987bbac7d8c99e6a88f3c1028609fbea6c73',replace_att='be89e623f6fd54d015f6a437ba9a34ffa39cc465',replace_rel='69d447e3fcb33817b866bd5688741f42bad7c26a',replace_obj='3b62823c003659737a68c3d78cfe721c57b44f5f')
SEED=20261122
PER=40
sha=lambda b:hashlib.sha256(b).hexdigest()
def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]
def fetch(url,path):
    if not path.exists():
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url,timeout=25) as resp: data=resp.read()
                path.write_bytes(data)
                break
            except OSError:
                if attempt==3: raise
                time.sleep(1+attempt)
    return path.read_bytes()
def image(name):
    assert name.startswith('coco/') and Path(name).name[:-4].isdigit()
    p=OLD/'images'/name
    raw=p.read_bytes() if p.is_file() else fetch('https://s3.amazonaws.com/images.cocodataset.org/val2017/'+Path(name).name,ROOT/'images'/Path(name).name)
    return raw
def hashes(name):
    raw=image(name)
    with Image.open(BytesIO(raw)) as im:
        y=np.asarray(ImageOps.exif_transpose(im).convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
    basis=np.cos(np.pi*(np.arange(32)+.5)*np.arange(8)[:,None]/32)
    a=(basis@y@basis.T).ravel()[1:]
    return sha(raw),int(''.join('1' if b else '0' for b in a>np.median(a)),2)
def main():
    (ROOT/'images').mkdir(exist_ok=True)
    older=jsonlines(OLD/'results/m0-20261102/manifest.jsonl')
    prior=jsonlines(OLD/'natural.jsonl')
    oldnames={n for r in older+prior for n in r['images'] if n.startswith('coco/')}
    older_audit=json.loads((OLD/'results/m0-20261102/image-audit.json').read_text())['files']
    recent={r['images'][0]:r['image_sha256'][0] for r in jsonlines(OLD/'runs/siglip-natural-full/predictions.jsonl')}
    previous={**{k:v['sha256'] for k,v in older_audit.items() if k.startswith('coco/')},**recent}
    assert set(previous)==oldnames
    hashes_seen=set();phashes=[]
    with ThreadPoolExecutor(8) as pool:
        for n,(digest,ph) in zip(sorted(oldnames),pool.map(hashes,sorted(oldnames))):
            assert previous[n]==digest
            hashes_seen.add(digest);phashes.append(ph)
    print('previous COCO bytes verified',len(oldnames),flush=True)
    used=set(oldnames); rng=random.Random(SEED); manifest=[];skips=[];cats={}
    for cat,oid in OIDS.items():
        data=fetch(f'https://raw.githubusercontent.com/Sri-Harsha/scpp/{PIN}/data/{cat}.json',ROOT/(cat+'.json'))
        assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==oid
        table=sorted(json.loads(data),key=lambda r:str(r['id']))
        rng.shuffle(table); added=[]
        source=iter(table)
        with ThreadPoolExecutor(8) as pool:
            while len(added)<PER:
                batch=[];scheduled=set()
                while len(batch)<min(12,PER-len(added)):
                    r=next(source)
                    name='coco/'+r['filename']
                    if name in used or name in scheduled:continue
                    txt=[r['caption'],r['caption2'],r['negative_caption']]
                    if not all(isinstance(t,str) and t.strip() for t in txt) or len(set(txt))!=3:
                        skips.append(dict(name=name,id=r['id'],category=cat,reason='missing_or_identical_captions'))
                        continue
                    batch.append((r,name,txt));scheduled.add(name)
                for (r,name,txt),(digest,ph) in zip(batch,pool.map(hashes,[b[1] for b in batch])):
                    if digest in hashes_seen or any((h^ph).bit_count()<=4 for h in phashes):
                        skips.append(dict(name=name,reason='bytes_or_phash'));used.add(name);continue
                    added.append(dict(id=f"scpp/{cat}/{r['id']}",cat=cat,img=name,sha256=digest,phash=f'{ph:016x}',texts=txt))
                    used.add(name); hashes_seen.add(digest);phashes.append(ph)
        assert len(added)==PER
        mix=random.Random(f'{SEED}/{cat}/wrong')
        mix.shuffle(added)
        offsets=mix.sample(range(1,PER),3)
        for j,r in enumerate(added): r['wrong']=[added[(j+k)%PER]['img'] for k in offsets]
        manifest.extend(added);cats[cat]=len(added)
        print('ready',cat,len(added),flush=True)
    random.Random('presentation/20261122').shuffle(manifest)
    out=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest).encode()
    target=ROOT/'manifest.private.jsonl'
    if target.exists(): assert target.read_bytes()==out
    else: target.write_bytes(out)
    spec=dict(seed=SEED,n=len(manifest),categories=cats,source_commit=PIN,source_blobs=OIDS,old_coco= len(oldnames),nov19_coco=len(recent),skips=skips,
              old_dedup='COCO id + SHA256 + 63-bit grayscale DCT hash Hamming <=4',manifest_sha256=sha(out),prepare_sha256=sha(Path(__file__).read_bytes()),
              date_utc=datetime.now(timezone.utc).isoformat(),wrong='3 fixed distinct same-category cyclic derangements, each image used once per arm',
              protocol_sha256=sha((ROOT/'protocol.txt').read_bytes()))
    p=ROOT/'freeze.json'
    if not p.exists():p.write_text(json.dumps(spec,indent=2))
    else:assert json.loads(p.read_text())['manifest_sha256']==sha(out)
    print('frozen',len(manifest),sha(out),flush=True)
if __name__=='__main__':main()
