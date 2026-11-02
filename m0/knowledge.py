"""Gold-blind, reproducible WordNet facts; identical facts in prose and triples."""
import re

STOP=set('a an the this that these those is are was were be been being has have had do does did of in on at to for from with and or as by it its he she they someone somebody person people'.split())


def retrieve(text,wn):
    words=[]
    for token in re.findall(r'[A-Za-z]+',text.lower()):
        if token in STOP or token in words or not wn.synsets(token,pos='n'): continue
        words.append(token)
        if len(words)==2: break
    facts=[]
    for word in words:
        for sense in wn.synsets(word,pos='n')[:2]:
            facts.append(dict(word=word,sense=sense.name(),definition=sense.definition(),
                lemmas=sorted(sense.lemma_names()),
                hypernyms=[dict(sense=h.name(),definition=h.definition())
                           for h in sorted(sense.hypernyms(),key=lambda h:h.name())[:2]]))
    return facts


def render(facts,arm):
    if arm=='K0':return ''
    lines=[]
    for f in facts:
        # Both arms preserve the same words, IDs, definitions, lemma order and directed facts.
        if arm=='K1':
            lines.append(f"The word {f['word']} has candidate sense {f['sense']}, defined as {f['definition']}. Its synonymous lemmas are {', '.join(f['lemmas'])}.")
            for h in f['hypernyms']:
                lines.append(f"Sense {f['sense']} is a kind of {h['sense']}, defined as {h['definition']}.")
        else:
            lines.append(f"word={f['word']}; sense={f['sense']}; definition={f['definition']}; synonyms={', '.join(f['lemmas'])}.")
            for h in f['hypernyms']:
                source,target=(h['sense'],f['sense']) if arm=='Kshuffle' else (f['sense'],h['sense'])
                lines.append(f"({source}, is_a, {target}); {h['sense']} definition={h['definition']}.")
    return '\n'.join(lines)


def prompt(hypothesis,knowledge):
    base=('Judge the hypothesis using only the available image evidence. '
          'entailment means the image supports the whole hypothesis; contradiction means '
          'the image provides evidence against it; neutral means the evidence is insufficient. '
          'Do not infer hidden intentions, identities, or events outside the image. '
          'Candidate dictionary senses may be irrelevant. Dictionary facts do not prove anything is visible. '
          'is_a means the source sense is a kind of the target sense, not the reverse.\n')
    if knowledge: base+='Dictionary knowledge:\n'+knowledge+'\n'
    return base+'Hypothesis: '+hypothesis+'\nAnswer exactly one label: entailment, contradiction, or neutral.'
