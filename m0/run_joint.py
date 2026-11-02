"""Frozen Qwen-VL seven-arm visual entailment pilot; no gold in model inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from .prepare import sha,save
from .run_dual import select_gpu,metadata
from .knowledge import retrieve,render,prompt

ARMS=('K0','K1','K2','Kshuffle','K0_no_image','K2_no_image','K2_replace_image')


def run(a):
    gpu=select_gpu(a.gpu)
    import torch
    import nltk
    from nltk.corpus import wordnet as wn
    from PIL import Image
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
    from exp5.models import require_gpu
    require_gpu(torch,'cuda:0');torch.manual_seed(2026);torch.set_num_threads(4)
    nltk.data.path.insert(0,a.wordnet)
    if wn.get_version()!='3.0':raise ValueError('WordNet version must be 3.0')
    manifest=Path(a.manifest); freeze=json.loads(manifest.with_name('freeze.json').read_text())
    if sha(manifest)!=freeze['manifest_sha256']:raise ValueError('Manifest changed')
    rows=[json.loads(x) for x in manifest.read_text().splitlines()]
    rows=[r for r in rows if r['benchmark']=='esnli']
    if a.split!='all':rows=[r for r in rows if r['split']==a.split]
    if a.limit:rows=rows[:a.limit]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    facts={r['sample_id']:retrieve(r['texts'][0],wn) for r in rows}
    save(out/'knowledge.private.json',facts)
    replacement={}
    for split in ('exploration','confirmation'):
        group=sorted([r for r in rows if r['split']==split],key=lambda r:r['sample_id'])
        for i,r in enumerate(group):replacement[r['sample_id']]=group[(i+1)%len(group)]['images'][0]
    start=time.perf_counter()
    processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True,use_fast=False,
        min_pixels=256*28*28,max_pixels=262144)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model_dir,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0').eval()
    meta=metadata(a.model_dir,manifest,gpu)
    meta.update(model_id='Qwen/Qwen2.5-VL-3B-Instruct',revision='66285546d2b821cf421d4f5eb2576359d3770cd3',
        dtype='bfloat16',split=a.split,max_new_tokens=16,do_sample=False,max_pixels=262144,
        min_pixels=256*28*28,wordnet_version=wn.get_version(),knowledge_sha256=sha(out/'knowledge.private.json'),
        wordnet_zip_sha256=sha(Path(a.wordnet)/'corpora/wordnet.zip'),
        model_load_seconds=time.perf_counter()-start,arms=list(a.arms),zero_paid_api_calls=True)
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    # Append only; an incomplete run is never silently summarized as complete.
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as log:
        for r in rows:
            for arm in a.arms:
                k='K0' if arm.startswith('K0') else ('K2' if arm.startswith('K2') else arm)
                knowledge=render(facts[r['sample_id']],k)
                text=prompt(r['texts'][0],knowledge)
                path=Path(a.images)/(replacement[r['sample_id']] if arm.endswith('replace_image') else r['images'][0])
                rec={x:r[x] for x in ('sample_id','benchmark','split','category','gold')}
                rec.update(model='qwen',arm=arm,prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    knowledge_tokens=len(processor.tokenizer(knowledge)['input_ids']),
                    knowledge_senses=len(facts[r['sample_id']]),
                    knowledge_edges=sum(len(f['hypernyms']) for f in facts[r['sample_id']]),
                    text_sha256=hashlib.sha256(r['texts'][0].encode()).hexdigest(),
                    image=None if arm.endswith('no_image') else path.relative_to(a.images).as_posix())
                if not arm.endswith('no_image') and not path.exists():
                    rec.update(status='unavailable',reason='missing_image',prediction='invalid')
                else:
                    content=[]; images=[]
                    if not arm.endswith('no_image'):
                        with Image.open(path) as im:images=[im.convert('RGB')]
                        content.append({'type':'image'});rec['image_sha256']=sha(path)
                    content.append({'type':'text','text':text})
                    messages=[{'role':'user','content':content}]
                    chat=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                    inputs=processor(text=[chat],images=images or None,return_tensors='pt',padding=True).to('cuda:0')
                    before=time.perf_counter()
                    with torch.inference_mode():
                        output=model.generate(**inputs,max_new_tokens=16,do_sample=False)
                    torch.cuda.synchronize()
                    ids=output[0,inputs['input_ids'].shape[1]:]
                    answer=processor.decode(ids,skip_special_tokens=True).strip()
                    pred=answer.lower().strip().rstrip('.')
                    if pred not in ('entailment','contradiction','neutral'):pred='invalid'
                    rec.update(status='ok',prediction=pred,raw_output=answer,
                        correct=int(pred==r['gold']),input_tokens=int(inputs['input_ids'].shape[1]),
                        output_tokens=len(ids),inference_seconds=time.perf_counter()-before)
                log.write(json.dumps(rec,allow_nan=False)+'\n');log.flush()
            print(r['sample_id'],'completed',flush=True)
    meta.update(inference_wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),n_registered=len(rows))
    save(out/'run.json',meta)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('manifest','images','model-dir','wordnet','out'):p.add_argument('--'+n,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--limit',type=int)
    p.add_argument('--split',choices=['all','exploration','confirmation'],default='all')
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    run(p.parse_args())
