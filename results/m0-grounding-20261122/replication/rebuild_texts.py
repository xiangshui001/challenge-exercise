"""Rebuild the two private manifests in a user-selected scratch directory."""
import argparse,hashlib,json,shutil,tempfile,urllib.request
from pathlib import Path

def sha(b): return hashlib.sha256(b).hexdigest()
def records(p): return [json.loads(x) for x in p.read_bytes().splitlines()]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--annotations-dir",type=Path)
    args=parser.parse_args()
    base=Path(__file__).resolve().parent
    dst=args.out.resolve()
    if dst.exists() or dst==base or base in dst.parents:
        parser.error("Choose a new directory outside this extracted package")
    dst.parent.mkdir(parents=True,exist_ok=True)
    pin=json.loads((base/"confirmation"/"freeze.json").read_bytes())
    database={}
    for cat,expected in pin["source_blobs"].items():
        if args.annotations_dir:
            b=(args.annotations_dir/(cat+".json")).read_bytes()
        else:
            url="https://raw.githubusercontent.com/Sri-Harsha/scpp/"+pin["source_commit"]+"/data/"+cat+".json"
            with urllib.request.urlopen(url,timeout=45) as response: b=response.read()
        if hashlib.sha1(("blob "+str(len(b))).encode()+bytes([0])+b).hexdigest()!=expected:
            raise ValueError("Upstream Git blob mismatch for "+cat)
        for r in json.loads(b):
            id="scpp/"+cat+"/"+str(r["id"])
            database[id]=("coco/"+r["filename"],[r["caption"],r["caption2"],r["negative_caption"]])
    with tempfile.TemporaryDirectory(dir=dst.parent) as scratch:
        stage=Path(scratch)/"run"
        for cohort in ("controls","confirmation","deeper"):
            shutil.copytree(base/cohort,stage/cohort)
        for cohort in ("controls","confirmation"):
            directory=stage/cohort
            freeze=json.loads((directory/"freeze.json").read_bytes()); rebuilt=[]
            extra=json.loads((directory/"manifest.phash.json").read_bytes()) if cohort=="confirmation" else {}
            for a in records(directory/"manifest.audit.jsonl"):
                i=a["id"]; cat=a.get("cat",a.get("category")); img=a.get("img",a.get("image"))
                filename,texts=database[i]
                if filename!=img or [sha(t.encode()) for t in texts]!=a["text_sha256"]: raise ValueError("Input mismatch at "+i)
                ph=a["phash"] if cohort=="controls" else extra[i]
                row=dict(id=i,cat=cat,img=img,sha256=a["sha256"],phash=ph,texts=texts,wrong=a["wrong"])
                if cohort=="confirmation":row.update(near=a["near"],far=a["far"])
                rebuilt.append(row)
            b=b"".join((json.dumps(x,ensure_ascii=False)+chr(10)).encode() for x in rebuilt)
            if sha(b)!=freeze["manifest_sha256"]: raise ValueError("Rebuilt manifest does not match freeze for "+cohort)
            (directory/"manifest.private.jsonl").write_bytes(b)
        stage.rename(dst)
    print("Both verified manifests are in",dst)

if __name__=="__main__":main()
