"""Frozen native dual encoders; raw captions, cosine ranks, private image inputs."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

from .prepare import sha,save
from .metrics import sugar,wsd

SPECS={
 'siglip':('google/siglip-base-patch16-224','7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed',64),
 'clip':('openai/clip-vit-base-patch32','3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',77)}


def select_gpu(index):
    if index not in (4,5): raise ValueError('Only user-assigned physical GPUs 4 and 5')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    ids={int(x.split(',')[0]):x.split(',')[1].strip() for x in raw.splitlines()}
    os.environ['CUDA_VISIBLE_DEVICES']=ids[index]
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    return ids[index]


def metadata(model_dir,manifest,gpu):
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),gpu_uuid=gpu,
        manifest_sha256=sha(manifest),code_sha256={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        packages={k:importlib.metadata.version(k) for k in ('torch','transformers','numpy','Pillow')},
        model_files={p.name:sha(p) for p in sorted(Path(model_dir).iterdir()) if p.is_file() and not p.name.endswith('.part')})


def run(a):
    gpu=select_gpu(a.gpu)
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel,AutoProcessor
    from exp5.models import require_gpu
    require_gpu(torch,'cuda:0')
    torch.manual_seed(2026); torch.set_num_threads(4)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    manifest=Path(a.manifest)
    freeze=json.loads(manifest.with_name('freeze.json').read_text())
    if sha(manifest)!=freeze['manifest_sha256']: raise ValueError('Manifest changed after registration')
    rows=[json.loads(x) for x in manifest.read_text().splitlines()]
    rows=[r for r in rows if r['benchmark'] in ('scpp','vwsd')]
    if a.split!='all': rows=[r for r in rows if r['split']==a.split]
    if a.limit: rows=rows[:a.limit]
    exclusions_path=manifest.with_name('exclusions.json')
    exclusions=json.loads(exclusions_path.read_text()) if exclusions_path.exists() else {}
    model_id,revision,max_len=SPECS[a.model]
    start=time.perf_counter()
    processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True,use_fast=False)
    model=AutoModel.from_pretrained(a.model_dir,local_files_only=True,torch_dtype=torch.float32,
                                   attn_implementation='sdpa').to('cuda:0').eval()
    meta=metadata(a.model_dir,manifest,gpu)
    meta.update(model_id=model_id,revision=revision,dtype='float32',split=a.split,
        model_load_seconds=time.perf_counter()-start,processor=processor.image_processor.to_dict(),
        text_max_length=max_len,text_padding='max_length' if a.model=='siglip' else True,
        lowercase=False,zero_paid_api_calls=True)
    files=sorted({i for r in rows for i in r['images']})
    images={}; hashes={}; missing={}
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    for name in files:
        p=Path(a.images)/name
        if not p.exists(): missing[name]='missing_image';continue
        with Image.open(p) as im:
            inputs=processor(images=im.convert('RGB'),return_tensors='pt').to('cuda:0')
        with torch.inference_mode(): f=model.get_image_features(**inputs).float()
        f=torch.nn.functional.normalize(f,dim=-1)[0].cpu().numpy()
        if not np.isfinite(f).all():raise ValueError('Nonfinite image features')
        images[name]=f;hashes[name]=sha(p)
    texts=list(dict.fromkeys(t for r in rows for t in r['texts']))
    feats={};lengths={}
    for offset in range(0,len(texts),32):
        batch=texts[offset:offset+32]
        for t,ids in zip(batch,processor.tokenizer(batch,truncation=False)['input_ids']): lengths[t]=len(ids)
        inputs=processor(text=batch,padding='max_length' if a.model=='siglip' else True,
                         max_length=max_len,truncation=True,return_tensors='pt').to('cuda:0')
        with torch.inference_mode(): f=model.get_text_features(**inputs).float()
        f=torch.nn.functional.normalize(f,dim=-1).cpu().numpy()
        if not np.isfinite(f).all(): raise ValueError('Nonfinite text features')
        feats.update(zip(batch,f))
    # Swap images only among the same benchmark AND split, using a deterministic cyclic permutation.
    replacements={}
    for split in ('exploration','confirmation'):
        group=sorted([r for r in rows if r['benchmark']=='scpp' and r['split']==split],key=lambda r:r['sample_id'])
        for i,r in enumerate(group): replacements[r['sample_id']]=group[(i+1)%len(group)]['images'][0]
    results=[]
    for r in rows:
        record={k:r[k] for k in ('sample_id','benchmark','split','category','gold','images')}
        record.update(model=a.model,image_sha256=[hashes.get(i) for i in r['images']],
            text_sha256=[__import__('hashlib').sha256(t.encode()).hexdigest() for t in r['texts']],
            token_lengths=[lengths[t] for t in r['texts']],truncated=sum(lengths[t]>max_len for t in r['texts']))
        if r['sample_id'] in exclusions:
            record.update(status='excluded',reason=exclusions[r['sample_id']]);results.append(record);continue
        if any(i not in images for i in r['images']):
            record.update(status='unavailable',reason='missing_image');results.append(record);continue
        tf=np.stack([feats[t] for t in r['texts']])
        im=np.stack([images[i] for i in r['images']])
        scores=(im@tf.T)
        if r['benchmark']=='scpp':
            ts=[float(tf[0]@tf[1]),float(tf[0]@tf[2]),float(tf[1]@tf[2])]
            record.update(status='ok',scores=scores[0].tolist(),text_scores=ts,metrics=sugar(scores[0],ts))
            replacement=replacements[r['sample_id']]
            if replacement in images:
                swapped=(images[replacement]@tf.T).tolist()
                record.update(replacement_image=replacement,replacement_scores=swapped,replacement_metrics=sugar(swapped))
        else:
            record.update(status='ok',scores=scores[:,0].tolist(),metrics=wsd(scores[:,0].tolist(),r['gold']))
        results.append(record)
    torch.cuda.synchronize()
    meta.update(inference_wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                n_registered=len(rows),n_valid=sum(r['status']=='ok' for r in results),missing_images=missing)
    meta['exclusions_sha256']=sha(exclusions_path) if exclusions_path.exists() else None
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in results))
    save(out/'run.json',meta)
    print(json.dumps({k:meta[k] for k in ('model_id','n_registered','n_valid','inference_wall_seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('manifest','images','model-dir','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--model',choices=list(SPECS),required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--split',choices=['exploration','confirmation','all'],default='all');p.add_argument('--limit',type=int)
    run(p.parse_args())
