"""Public-source acquisition helpers. Images/weights stay outside version control."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
import zipfile
from .prepare import sha,save

MODELS={
 'siglip':('google/siglip-base-patch16-224','7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed'),
 'clip':('openai/clip-vit-base-patch32','3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'),
 'qwen':('Qwen/Qwen2.5-VL-3B-Instruct','66285546d2b821cf421d4f5eb2576359d3770cd3')}


def fetch(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return
    temp=path.with_suffix(path.suffix+'.partial')
    with urllib.request.urlopen(url,timeout=60) as r,temp.open('wb') as f:shutil.copyfileobj(r,f)
    temp.replace(path)


def acquire_images(manifest,out,flickr_root=None,flickr_archive=None,vwsd_archive=None):
    rows=[json.loads(x) for x in Path(manifest).read_text().splitlines()]
    names={x for r in rows for x in r['images']};out=Path(out)
    coco=sorted(x for x in names if x.startswith('coco/'))
    # Official COCO download endpoint uses HTTP; record hashes and decode audit afterwards.
    with concurrent.futures.ThreadPoolExecutor(4) as p:
        list(p.map(lambda x:fetch('http://images.cocodataset.org/val2017/'+Path(x).name,out/x),coco))
    wanted={Path(x).name for x in names if x.startswith('flickr/')}
    if flickr_root:
        for name in sorted(wanted):
            target=out/'flickr'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(flickr_root)/name,target)
    elif flickr_archive:
        with tarfile.open(flickr_archive,'r:*') as tar:
            for member in tar:
                name=Path(member.name).name
                if member.isfile() and name in wanted:
                    target=out/'flickr'/name;target.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
    if vwsd_archive:
        wanted={Path(x).name for x in names if x.startswith('vwsd/')}
        with zipfile.ZipFile(vwsd_archive) as z:
            for name in z.namelist():
                filename=Path(name).name
                if filename in wanted:
                    target=out/'vwsd'/filename;target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(z.read(name))
    return [x for x in sorted(names) if not (out/x).exists()]


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    m=sub.add_parser('model');m.add_argument('name',choices=MODELS);m.add_argument('--out',required=True)
    w=sub.add_parser('wordnet');w.add_argument('--out',required=True)
    i=sub.add_parser('images')
    for x in ('manifest','out'):i.add_argument('--'+x,required=True)
    for x in ('flickr-root','flickr-archive','vwsd-archive'):i.add_argument('--'+x)
    a=p.parse_args()
    if a.command=='model':
        from huggingface_hub import snapshot_download
        repo,revision=MODELS[a.name]
        snapshot_download(repo_id=repo,revision=revision,local_dir=a.out,
            allow_patterns=['*.json','*.txt','*.model','*.safetensors','README.md','LICENSE']+
                           (['pytorch_model.bin'] if a.name=='clip' else []))
    elif a.command=='wordnet':
        target=Path(a.out)/'corpora/wordnet.zip'
        fetch('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip',target)
        print('WordNet archive SHA256:',sha(target))
    else:
        missing=acquire_images(a.manifest,a.out,a.flickr_root,a.flickr_archive,a.vwsd_archive)
        print(json.dumps(dict(missing=missing)))
        if missing:raise SystemExit(2)


if __name__=='__main__':main()
