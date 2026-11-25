"""Fail closed on incomplete public result sets or mismatched frozen identities."""
import argparse
import json
from pathlib import Path
from .prepare import sha,save

WEIGHTS={
 'siglip':{'model.safetensors':'2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8'},
 'clip':{'pytorch_model.bin':'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'},
 'qwen':{'model-00001-of-00002.safetensors':'41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8',
         'model-00002-of-00002.safetensors':'365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4'}}
ARMS={'K0','K1','K2','Kshuffle','K0_no_image','K2_no_image','K2_replace_image'}


def validate(root):
    root=Path(root);freeze=json.loads((root/'freeze.json').read_text())
    if sha(root/'manifest.jsonl')!=freeze['public_manifest_sha256']:raise ValueError('Public manifest hash mismatch')
    manifest={r['sample_id']:r for r in map(json.loads,(root/'manifest.jsonl').read_text().splitlines())}
    exclusions=json.loads((root/'exclusions.json').read_text())
    audit=json.loads((root/'image-audit.json').read_text())
    if audit['missing'] or audit['active_byte_duplicate_groups']:raise ValueError('Image audit incomplete')
    if audit['manifest_sha256']!=freeze['manifest_sha256']:raise ValueError('Image audit identity mismatch')
    counts={}
    for model in WEIGHTS:
        meta=json.loads((root/f'{model}-run.json').read_text())
        if meta['manifest_sha256']!=freeze['manifest_sha256']:raise ValueError('Run manifest mismatch')
        if model!='qwen' and meta['exclusions_sha256']!=sha(root/'exclusions.json'):
            raise ValueError('Exclusion bytes differ from the actual run')
        if any(meta['model_files'].get(k)!=v for k,v in WEIGHTS[model].items()):raise ValueError('Model weights mismatch')
        selected={i for i,r in manifest.items() if (r['benchmark']=='esnli')==(model=='qwen')}
        expected={(i,a) for i in selected for a in (ARMS if model=='qwen' else {'K0'})}
        actual=set();ok=0
        for r in map(json.loads,(root/f'{model}-predictions.jsonl').read_text().splitlines()):
            i=r['sample_id'];key=(i,r.get('arm','K0'))
            if key in actual:raise ValueError('Duplicate result')
            actual.add(key);source=manifest[i]
            if any(r[k]!=source[k] for k in ('gold','benchmark','split','category')):raise ValueError('Gold/group mismatch')
            hashes=r['text_sha256'] if isinstance(r['text_sha256'],list) else [r['text_sha256']]
            if hashes!=source['text_sha256']:raise ValueError('Text identity mismatch')
            if i in exclusions:
                if r['status']!='excluded':raise ValueError('Quarantined unit used')
                continue
            if r['status']!='ok':raise ValueError('Unavailable result; report partial instead of complete')
            ok+=1
            names=r['images'] if model!='qwen' else ([r['image']] if r['image'] else [])
            hs=r['image_sha256'] if model!='qwen' else ([r['image_sha256']] if r['image'] else [])
            if len(names)!=len(hs):raise ValueError('Incomplete image hashes')
            if model!='qwen' and names!=source['images']:raise ValueError('Candidate ordering mismatch')
            if model=='qwen' and names and not r['arm'].endswith('replace_image') and names!=source['images']:
                raise ValueError('Original image changed')
            if model=='qwen' and r['arm'].endswith('no_image') and names:raise ValueError('No-image control has image')
            if any(audit['files'][n]['sha256']!=h for n,h in zip(names,hs)):raise ValueError('Image identity mismatch')
        if actual!=expected:raise ValueError('Missing or extra predictions')
        if meta['n_registered']!=len(selected):raise ValueError('Run coverage mismatch')
        counts[model]=dict(rows=len(actual),ok=ok)
    return dict(status='passed',counts=counts,public_files_sha256={p.name:sha(p) for p in sorted(root.iterdir())
        if p.is_file() and p.name not in ('validation.json','README.md')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args()
    result=validate(a.root);save(Path(a.root)/'validation.json',result);print(result['status'],result['counts'])
