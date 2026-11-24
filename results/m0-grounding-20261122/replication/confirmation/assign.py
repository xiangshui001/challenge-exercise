import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from prepare_confirm import ROOT,OLD,locate,encode,sha
sys.path.insert(0,str(OLD))
from followup.run import idle_gpu
from m0.prepare import sha as filehash
def main():
    p=ROOT/'base.private.jsonl'; data=p.read_bytes()
    audit=json.loads((ROOT/'base.audit.json').read_text());assert audit['sha256']==sha(data)
    rows=[json.loads(l) for l in data.splitlines()]
    path=OLD/'models/siglip'
    meta=json.loads((OLD/'runs/siglip-natural-full/run.json').read_text())
    assert all(filehash(path/k)==v for k,v in meta['model_files'].items())
    idle_gpu(0)
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModel
    proc=AutoProcessor.from_pretrained(path,local_files_only=True,use_fast=False)
    torch.set_num_threads(3)
    model=AutoModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda:0').eval()
    vectors=[];sims={}
    with torch.inference_mode():
        for start in range(0,len(rows),16):
            images=[]
            for r in rows[start:start+16]:
                assert filehash(locate(r['img']))==r['sha256']
                with Image.open(locate(r['img'])) as im:images.append(im.convert('RGB'))
            x=model.get_image_features(**proc(images=images,return_tensors='pt').to('cuda:0'))
            vectors.extend(torch.nn.functional.normalize(x if hasattr(x,'ndim') else x.pooler_output,dim=-1).cpu().numpy())
    for cat in sorted(set(r['cat'] for r in rows)):
        ix=[i for i,r in enumerate(rows) if r['cat']==cat]; sub=[rows[i] for i in ix]; idx={r['img']:i for i,r in enumerate(sub)}
        V=np.stack([vectors[i] for i in ix]);S=V@V.T; blocked=np.eye(len(sub),dtype=bool)
        for i,r in enumerate(sub):
            for n in r['wrong']:blocked[i,idx[n]]=True
        for arm,sign in [('near',-1),('far',1)]:
            cost=np.where(blocked,1e5,sign*S.astype('float64'))
            a,b=linear_sum_assignment(cost);assert list(a)==list(range(30)) and not blocked[a,b].any()
            for i,j in zip(a,b):sub[i][arm]=sub[j]['img']
            blocked[a,b]=True
        for i,r in enumerate(sub):
            sims[r['id']]={arm:float(S[i,idx[n]]) for arm,n in zip(['wrong1','wrong2','wrong3','near','far'],[*r['wrong'],r['near'],r['far']])}
            assert len(set([r['img'],*r['wrong'],r['near'],r['far']]))==6
    final=encode(rows);(ROOT/'manifest.private.jsonl').write_bytes(final)
    (ROOT/'similarities.json').write_text(json.dumps(sims,indent=1))
    freeze={**audit,'manifest_sha256':sha(final),'protocol_sha256':filehash(ROOT/'protocol.txt'),'assign_sha256':filehash(Path(__file__)),'prepare_sha256':filehash(ROOT/'prepare_confirm.py'),'infer_sha256':filehash(ROOT/'infer.py'),'similarities_sha256':filehash(ROOT/'similarities.json'),'selector_files':meta['model_files']}
    (ROOT/'freeze.json').write_text(json.dumps(freeze,indent=2));print('all assignments frozen',sha(final),flush=True)
if __name__=='__main__':main()
