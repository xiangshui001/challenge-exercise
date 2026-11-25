"""Audit prompt identities from private facts without publishing dictionary texts."""
import argparse,hashlib,json
from pathlib import Path
from .knowledge import render,prompt
from .prepare import save,sha


def verify(manifest,knowledge,predictions):
    rows={r['sample_id']:r for r in map(json.loads,Path(manifest).read_text().splitlines())}
    facts=json.loads(Path(knowledge).read_text());records=list(map(json.loads,Path(predictions).read_text().splitlines()))
    lookup={};checked=0
    for r in records:
        arm=r['arm'];k='K0' if arm.startswith('K0') else ('K2' if arm.startswith('K2') else arm)
        text=prompt(rows[r['sample_id']]['texts'][0],render(facts[r['sample_id']],k))
        if hashlib.sha256(text.encode()).hexdigest()!=r['prompt_sha256']:raise ValueError('Prompt reconstruction mismatch')
        lookup[(r['sample_id'],arm)]=r;checked+=1
    ids=sorted(facts)
    different=sum(lookup[i,'K2']['prompt_sha256']!=lookup[i,'Kshuffle']['prompt_sha256'] for i in ids)
    same=sum(lookup[i,'K2']['prediction']==lookup[i,'Kshuffle']['prediction'] for i in ids)
    return dict(checked_prompts=checked,units=len(ids),distinct_K2_corrupted_prompts=different,
        identical_K2_corrupted_predictions=same,zero_sense_units=sum(not facts[i] for i in ids),
        zero_edge_units=sum(not any(f['hypernyms'] for f in facts[i]) for i in ids),
        private_knowledge_sha256=sha(knowledge),retrieval_relevance='not human validated; coverage is not correctness',
        knowledge_information_control='same sense IDs, definitions, lemmas, edge endpoints and direction in K1/K2; rendering differs')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('manifest','knowledge','predictions','out'):p.add_argument('--'+n,required=True)
    a=p.parse_args();save(a.out,verify(a.manifest,a.knowledge,a.predictions))
