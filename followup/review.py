"""Build a local anonymous review form or compare a later export to the fixed AI review."""
import argparse
import json
from pathlib import Path
from collections import Counter

HTML = """<!doctype html><meta charset="UTF-8"><html lang="zh"><title>M0 图像复核</title>
<style>body{max-width:1000px;margin:2rem auto;font:18px/1.5 system-ui;background:#fafafa;color:#182029} img{max-width:100%;max-height:480px;object-fit:contain} figure{margin:.3rem;display:inline-grid} section{background:white;padding:1rem} textarea{width:95%;height:4rem}button,select,input{font:inherit;margin:.3rem}pre{white-space:pre-wrap}label{display:block}</style>
<h1>M0 图像复核</h1><p>先看图和原文，独立填写判断及证据。可以跳题。请导出 JSON 保存进度，下次导入可继续。</p>
<select id="nav"></select><button id="prev">上一题</button><button id="next">下一题</button><section id="panel"></section>
<button id="save">导出 JSON</button><label>导入上次记录 <input type="file" id="upload" accept=".json"></label><p id="status" role="status"></p>
<script>const items=QUEUE, answers={}; const el=id=>document.getElementById(id);
const add=(type,host,txt)=>{let n=document.createElement(type);if(txt!==undefined)n.textContent=txt;host.append(n);return n};
items.forEach(q=>{let o=add('option',el('nav'),q.id);o.value=q.id});
function draw(){const q=items[el('nav').selectedIndex],p=el('panel');p.replaceChildren();const a=answers[q.id]??={};
q.images.forEach((src,i)=>{const f=add('figure',p);add('figcaption',f,q.images.length>1?'候选 '+i:'图像');let im=add('img',f);im.src=src;if(q.images.length>1)im.width=180});
const options=q.task==='scpp'?['T','F','partial','uncertain']:q.task==='esnli'?['entailment','contradiction','neutral']:q.images.map((_,i)=>String(i));
q.texts.forEach((t,i)=>{const l=add('label',p,t);const s=add('select',l);['',...options].forEach(x=>{let o=add('option',s,x||'待填');o.value=x});s.value=(a.judgments||[])[i]||'';s.onchange=()=>{(a.judgments??=[])[i]=s.value}});
add('p',p,q.task==='scpp'?'逐句判断，T 支持，F 有反证，partial 部分支持，uncertain 无法判断。请记录前后描述是否等义、是否有多个合理读法。':q.task==='esnli'?'记录可见的支持或反证，以及哪些细节仅凭图像无法判断。':'选择最符合短语的候选图，注明其他可能答案。');
if(q.candidate_senses){let d=add('details',p);add('summary',d,'候选 WordNet 义项与上位关系');q.candidate_senses.forEach(f=>{let l=add('label',d);let c=add('input',l);c.type='checkbox';c.checked=(a.senses||[]).includes(f.sense);l.append(f.sense+' — '+f.definition+' → '+f.hypernyms.map(h=>h.sense).join(', '));c.onchange=()=>{let set=new Set(a.senses||[]);c.checked?set.add(f.sense):set.delete(f.sense);a.senses=[...set]}})}
add('label',p,'理由、视觉证据、歧义或候选库遗漏');let n=add('textarea',p);n.value=a.evidence||'';n.oninput=()=>a.evidence=n.value}
el('nav').onchange=draw; for(const [id,delta] of [['prev',-1],['next',1]])el(id).onclick=()=>{el('nav').selectedIndex=Math.max(0,Math.min(items.length-1,el('nav').selectedIndex+delta));draw()};
el('save').onclick=()=>{let a=document.createElement('a');let u=URL.createObjectURL(new Blob([JSON.stringify({version:1,answers},null,2)],{type:'application/json'}));a.href=u;a.download='human-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
el('upload').onchange=async e=>{try{let r=JSON.parse(await e.target.files[0].text());if(r.version!==1||!r.answers||Object.keys(r.answers).some(k=>!items.some(q=>q.id===k)))throw Error();Object.assign(answers,r.answers);draw();el('status').textContent='已导入'}catch{el('status').textContent='无法识别此记录'}};draw();</script>"""


def build(queue):
    raw = json.loads(queue.read_text(encoding="utf8"))
    safe = json.dumps(raw, ensure_ascii=True).replace("<", "\\u003c")
    queue.with_name("review.html").write_text(HTML.replace("QUEUE", safe), encoding="utf8")


def compare(export, reference):
    human = json.loads(export.read_text(encoding="utf8"))["answers"]
    senses = json.loads(reference.with_name("selected-senses.json").read_text())
    counts = Counter()
    rows = []
    for line in reference.read_text(encoding="utf8").splitlines():
        r = json.loads(line)
        response = human.get(r["id"], {})
        judgments = response.get("judgments", [])
        filled = len(judgments)==len(r["text_order"]) and all(judgments)
        match = (",".join(judgments)==r["answer"]) if filled else None
        counts["pending" if match is None else "agree" if match else "disagree"] += 1
        rows.append(dict(id=r["id"], sample_id=r["sample_id"], ai=r["answer"], human=judgments,
                         agree=match, human_notes=response.get("evidence"),human_senses=response.get("senses"),
                         ai_senses=senses.get(r["id"])))
    return dict(counts=dict(counts), rows=rows)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--queue",type=Path)
    p.add_argument("--human",type=Path)
    p.add_argument("--reference",type=Path)
    a=p.parse_args()
    if a.queue:
        build(a.queue)
    else:
        print(json.dumps(compare(a.human,a.reference),ensure_ascii=False,indent=2))

