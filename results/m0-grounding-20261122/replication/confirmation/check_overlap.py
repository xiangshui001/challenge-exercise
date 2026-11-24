import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
OLD=Path('/root/m0-followup-20261119')
DEV=Path('/root/challenge-new-controls-20261122-a')
def rows(p):return [json.loads(l) for l in p.read_text().splitlines()]
def checks(texts):return tuple(hashlib.sha256(x.encode()).hexdigest() for x in texts)
def main():
    earlier=[tuple(r['text_sha256']) for r in rows(OLD/'results/m0-20261102/manifest.jsonl') if r['benchmark']=='scpp']
    earlier.extend(checks(r['texts']) for r in rows(OLD/'natural.jsonl') if r['benchmark']=='scpp')
    earlier.extend(checks(r['texts']) for r in rows(DEV/'manifest.private.jsonl'))
    assert len(earlier)==820
    known={h for group in earlier for h in group}
    new=rows(R/'manifest.private.jsonl')
    current=[checks(r['texts']) for r in new]
    report=dict(prior_groups=len(earlier),new_groups=len(current),matches_by_position=[sum(t[i] in known for t in current) for i in range(3)],
        any_sentence_match=sum(any(h in known for h in t) for t in current),ordered_triples_match=sum(t in set(earlier) for t in current),
        unordered_triples_match=sum(tuple(sorted(t)) in {tuple(sorted(o)) for o in earlier} for t in current),new_unique_caption_hashes=len(set(h for t in current for h in t)))
    (R/'caption_overlap.json').write_bytes(json.dumps(report,indent=2).encode())
    print(report)
if __name__=='__main__':main()
