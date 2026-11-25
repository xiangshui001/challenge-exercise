import unittest
from m0.metrics import sugar,wsd,classification,paired_bootstrap
from m0.knowledge import render,prompt
from m0.prepare import sample_strata
import random


class M0Metrics(unittest.TestCase):
    def test_strict_sugar_and_official_tot(self):
        r=sugar([.8,.7,.5],[.6,.4,.5])
        self.assertEqual((r['itt'],r['tot'],r['xor']),(1,1,0))
        self.assertEqual(sugar([.8,.5,.5])['xor'],1)
        self.assertEqual(sugar([.5,.5,.5])['both_wrong'],1)
        self.assertEqual(sugar([.8,.7,.5],[.6,.4,.6])['tot'],0)

    def test_wsd_candidate_count_and_ties(self):
        r=wsd([1,1,0,0,0,0,0,0,0,0],1)
        self.assertEqual((r['top1'],r['mrr'],r['top_tie']),(0,.5,1))
        with self.assertRaises(ValueError):wsd([1,2],0)

    def test_invalid_prediction_keeps_denominator(self):
        r=classification(['entailment','neutral','contradiction'],['entailment','entailment','invalid'])
        self.assertEqual(r['accuracy'],1/3)
        self.assertEqual(r['neutral_false_support'],1)
        self.assertEqual(r['invalid'],1)
        self.assertAlmostEqual(r['macro_f1'],(2/3)/3)

    def test_paired_no_change(self):
        r=paired_bootstrap([0,1,1],[0,1,1])
        self.assertEqual((r['estimate'],r['low'],r['high']),(0,0,0))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):sugar([float('nan'),1,0])
        with self.assertRaises(ValueError):paired_bootstrap([])

    def test_unique_images_across_strata(self):
        strata={k:[dict(sample_id=k+str(i),images=[str(i)]) for i in range(10)] for k in ('a','b')}
        rows=sample_strata(strata,4,2,random.Random(2026))
        self.assertEqual(len({r['images'][0] for r in rows}),8)
        self.assertEqual(sum(r['split']=='exploration' for r in rows),4)

    def test_equal_fact_content_and_corruption(self):
        facts=[dict(word='dog',sense='dog.n.01',definition='a domestic canine',lemmas=['dog','domestic_dog'],
               hypernyms=[dict(sense='canine.n.02',definition='a canid')])]
        a,b,c=[render(facts,k) for k in ('K1','K2','Kshuffle')]
        for item in ('dog.n.01','a domestic canine','domestic_dog','canine.n.02','a canid'):
            self.assertIn(item,a);self.assertIn(item,b)
        self.assertIn('(dog.n.01, is_a, canine.n.02)',b)
        self.assertIn('(canine.n.02, is_a, dog.n.01)',c)
        self.assertNotIn('gold',prompt('a dog',b))


if __name__=='__main__':unittest.main()
