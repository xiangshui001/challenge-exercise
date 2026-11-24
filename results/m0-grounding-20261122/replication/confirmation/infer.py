from pathlib import Path
import argparse, collections, hashlib, json, os, sys, time
from datetime import datetime, timezone
from importlib.metadata import version

ROOT=Path(__file__).resolve().parent
OLD=Path('/root/m0-followup-20261119')
sys.path.insert(0,str(OLD))
from followup.run import idle_gpu
from m0.run_dual import SPECS
from m0.prepare import sha

ARMS=['original','wrong1','wrong2','wrong3','near','far','text']
PROMPT='Is the following caption an accurate description of the image? Answer Yes or No. Caption: {caption}'
def stamp():return datetime.now(timezone.utc).isoformat()
def digest(data):return hashlib.sha256(data).hexdigest()
def image_path(name):
    from prepare_confirm import locate
    return locate(name)
def image_rgb(name):
    from PIL import Image
    with Image.open(image_path(name)) as im:return im.convert('RGB')
def get_features(t):
    return t if hasattr(t,'ndim') else t.pooler_output
def dual(modelname, allrows, todo, dest):
    import torch
    from transformers import AutoModel,AutoProcessor
    path=OLD/'models'/modelname
    _,_,maxlen=SPECS[modelname]
    processor=AutoProcessor.from_pretrained(path,local_files_only=True,use_fast=False)
    model=AutoModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda:0').eval()
    imgs=sorted({n for r in todo for n in [r['img'],*r['wrong'],r['near'],r['far']]})
    texts=sorted({t for r in todo for t in r['texts']})
    vf,tf={},{}
    trunc=0
    with torch.inference_mode():
        for i in range(0,len(imgs),16):
            names=imgs[i:i+16]
            inputs=processor(images=[image_rgb(n) for n in names],return_tensors='pt').to('cuda:0')
            mat=torch.nn.functional.normalize(get_features(model.get_image_features(**inputs)).float(),dim=-1)
            vf.update(zip(names,mat.cpu()))
        for i in range(0,len(texts),32):
            batch=texts[i:i+32]
            trunc+=sum(len(row)>maxlen for row in processor.tokenizer(batch,truncation=False)['input_ids'])
            inputs=processor(text=batch,padding='max_length' if modelname=='siglip' else True,
                             max_length=maxlen,truncation=True,return_tensors='pt').to('cuda:0')
            mat=torch.nn.functional.normalize(get_features(model.get_text_features(**inputs)).float(),dim=-1)
            tf.update(zip(batch,mat.cpu()))
        for r in todo:
            features=torch.stack([tf[t] for t in r['texts']])
            scores={arm:(features@vf[n]).tolist() for arm,n in zip(ARMS,[r['img'],*r['wrong'],r['near'],r['far']])}
            sim=features@features.T
            scores['text']=[float((sim[i,(i+1)%3]+sim[i,(i+2)%3])/2) for i in range(3)]
            emit(dest,r,scores)
    return {'max_length':maxlen,'truncated_unique_captions_this_pass':trunc}

def emit(dest,r,scores,**extra):
    assert list(scores)==ARMS and all(len(vals)==3 for vals in scores.values())
    dest.write(json.dumps({'id':r['id'],'cat':r['cat'],'sha256':r['sha256'],'wrong':r['wrong'],'near':r['near'],'far':r['far'],
        'text_sha256':[digest(t.encode()) for t in r['texts']],'scores':scores,**extra},allow_nan=False)+'\n')
    dest.flush()

def qwen(allrows,todo,dest,smoke):
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    path=OLD/'models/qwen'
    proc=AutoProcessor.from_pretrained(path,local_files_only=True,use_fast=False,
                                        min_pixels=256*28*28,max_pixels=262144)
    proc.tokenizer.padding_side='left'
    labels=[proc.tokenizer.encode(x,add_special_tokens=False) for x in ['Yes','No']]
    assert all(len(t)==1 for t in labels) and labels[0]!=labels[1]
    ids=[t[0] for t in labels]
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,
               attn_implementation='sdpa').to('cuda:0').eval()
    weight=model.lm_head.weight[ids].float()
    def score(captions, image,check=False):
        chats=[]
        for t in captions:
            text=PROMPT.format(caption=t)
            parts=([{'type':'image'}] if image is not None else [])+[{'type':'text','text':text}]
            chats.append(proc.apply_chat_template([{'role':'user','content':parts}],tokenize=False,add_generation_prompt=True))
        inputs=proc(text=chats,images=[image]*len(chats) if image is not None else None,
                    return_tensors='pt',padding=True).to('cuda:0')
        assert ('pixel_values' in inputs)==(image is not None)
        assert (inputs.input_ids[:,-1]==inputs.input_ids[0,-1]).all() and (inputs.attention_mask[:,-1]==1).all()
        with torch.inference_mode():
            h=model.model(**inputs,return_dict=True,use_cache=False).last_hidden_state[:,-1,:].float()
            logits=torch.nn.functional.linear(h,weight)
            if check:
                direct=model(**inputs,return_dict=True,use_cache=False,logits_to_keep=1).logits[:,-1,ids].float()
                assert (direct-logits).abs().max().item()<0.25
            return (logits[:,0]-logits[:,1]).cpu().tolist()
    for k,r in enumerate(todo):
        scores={}
        for arm,name in zip(ARMS,[r['img'],*r['wrong'],r['near'],r['far'],None]):
            image=image_rgb(name) if name is not None else None
            scores[arm]=score(r['texts'],image,check=smoke and k==0 and arm in ('text','original'))
        emit(dest,r,scores)
        if (k+1)%10==0:print('qwen completed in this pass',k+1,'/',len(todo),flush=True)
    return {'token_ids':ids,'prompt':PROMPT,'score':'float32 projection of final BF16 hidden state onto the frozen Yes and No output weights; subtract','min_pixels':256*28*28,'max_pixels':262144}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=['qwen','siglip','clip'])
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--gpu',type=int,default=0)
    a=parser.parse_args(); model=a.model; run=('smoke_' if a.smoke else 'full_')+model
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    freeze=json.loads((ROOT/'freeze.json').read_text()); manifest=ROOT/'manifest.private.jsonl'
    assert freeze['manifest_sha256']==sha(manifest) and freeze['protocol_sha256']==sha(ROOT/'protocol.txt')
    assert freeze['infer_sha256']==sha(Path(__file__))
    rows=[json.loads(l) for l in manifest.read_text().splitlines()]
    assert len(rows)==150 and len({r['sha256'] for r in rows})==150
    for r in rows: assert sha(image_path(r['img']))==r['sha256']
    hashes={r['img']:r['sha256'] for r in rows}
    for r in rows: assert len({r['sha256'],*(hashes[x] for x in [*r['wrong'],r['near'],r['far']])})==6
    old_meta=json.loads((OLD/'runs'/(model+'-natural-full')/'run.json').read_text())
    for file,expected in old_meta['model_files'].items(): assert sha(OLD/'models'/model/file)==expected
    source=sha(Path(__file__)); identity={'infer_sha256':source,'manifest_sha256':sha(manifest),'model_files':old_meta['model_files'],'protocol_sha256':freeze['protocol_sha256']}
    target=ROOT/(run+'.jsonl'); ident=ROOT/(run+'.identity.json')
    if ident.exists():assert json.loads(ident.read_text())==identity
    else:ident.write_text(json.dumps(identity))
    chosen=rows[:2] if a.smoke else rows
    done=[]
    if target.exists():
        with target.open('r+b') as f:
            while True:
                start=f.tell(); line=f.readline()
                if not line:break
                if not line.endswith(b'\n'):f.truncate(start);break
                done.append(json.loads(line)['id'])
    assert done==[r['id'] for r in chosen[:len(done)]]
    todo=chosen[len(done):]
    if not todo and (ROOT/(run+'.run.json')).is_file():
        print(run,'already sealed',flush=True);return
    gpu=idle_gpu(a.gpu)
    import torch
    torch.manual_seed(20261122);torch.set_num_threads(3)
    start=time.monotonic()
    print('start',run,'remaining',len(todo),flush=True)
    with target.open('a') as f:
        meta=qwen(rows,todo,f,a.smoke) if model=='qwen' else dual(model,rows,todo,f)
    out={'model':model,'run':run,'gpu_model':torch.cuda.get_device_name(0),'gpu_uuid_sha256':digest(gpu.encode()),
         'created_utc':stamp(),'n':len(chosen),'resumed_count':len(done),'wall_seconds_last_pass':time.monotonic()-start,
         'predictions_sha256':sha(target),'peak_allocated':torch.cuda.max_memory_allocated(),
         'packages':{x:version(x) for x in ('torch','transformers','Pillow','numpy')},**identity,**meta}
    (ROOT/(run+'.run.json')).write_text(json.dumps(out,indent=2));print('sealed',run,len(chosen),flush=True)
if __name__=='__main__':main()
