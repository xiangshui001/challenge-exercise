import math
import unittest
import numpy as np

from exp5.metrics import evaluate, rank_scores, weighted_rrf
from exp5.stats import bootstrap, changes


class MetricsTests(unittest.TestCase):
    def rows(self, rank=1, positive_count=1):
        ids = [f"c{i:02}" for i in range(60)]
        scores = np.arange(60, 0, -1, dtype=float)[None, :]
        positives = {"q": set(ids[rank - 1:rank - 1 + positive_count])}
        return evaluate(scores, ["q"], ids, positives, ["image"])

    def test_positive_first(self):
        r = self.rows()[0]
        self.assertEqual(r["hit_recall_at_1"], 1)
        self.assertEqual(r["ndcg_at_10"], 1)

    def test_positive_seventh(self):
        r = self.rows(7)[0]
        self.assertEqual(r["hit_recall_at_5"], 0)
        self.assertEqual(r["hit_recall_at_10"], 1)
        self.assertAlmostEqual(r["ndcg_at_10"], 1 / math.log2(8))

    def test_five_positives_one_hit(self):
        ids = [f"c{i:02}" for i in range(60)]
        positives = {"q": {ids[i] for i in (0, 20, 21, 22, 23)}}
        r = evaluate(np.arange(60, 0, -1)[None, :], ["q"], ids, positives, ["image"])[0]
        self.assertEqual(r["hit_recall_at_10"], 1)
        self.assertEqual(r["set_recall_at_10"], 0.2)
        self.assertAlmostEqual(r["ndcg_at_10"], 1 / sum(1 / math.log2(j + 1) for j in range(1, 6)))

    def test_ideal_five_positive_ndcg(self):
        self.assertAlmostEqual(self.rows(1, 5)[0]["ndcg_at_10"], 1)

    def test_tie_uses_lexical_id_not_input_order(self):
        self.assertEqual(rank_scores(np.ones((1, 3)), ["z", "b", "a"]).tolist(), [[2, 1, 0]])

    def test_invalid_inputs(self):
        scenarios = [(np.ones((1, 2)), ["a", "a"], {"q": {"a"}}),
                     (np.ones((1, 2)), ["a", "b"], {"q": {"missing"}}),
                     (np.ones((1, 2)), ["a", "b"], {"q": set()}),
                     (np.ones((1, 3)), ["a", "b"], {"q": {"a"}}),
                     (np.array([[np.nan, 0]]), ["a", "b"], {"q": {"a"}}),
                     (np.array([[np.inf, 0]]), ["a", "b"], {"q": {"a"}})]
        for scores, ids, positives in scenarios:
            with self.subTest(scores=scores, ids=ids), self.assertRaises(ValueError):
                evaluate(scores, ["q"], ids, positives, ["image"])
        with self.assertRaises(ValueError):
            evaluate(np.ones((2, 2)), ["q"], ["a", "b"], {"q": {"a"}}, ["image"])

    def test_ndcg_against_sklearn(self):
        try:
            from sklearn.metrics import ndcg_score
        except ImportError:
            self.skipTest("Optional scikit-learn oracle is not installed")
        rng = np.random.default_rng(124)
        ids = [str(i) for i in range(60)]
        for count in (1, 5):
            for _ in range(10):
                scores = rng.normal(size=(1, 60))
                pos = rng.choice(60, count, replace=False)
                truth = np.zeros((1, 60))
                truth[0, pos] = 1
                result = evaluate(scores, ["q"], ids, {"q": {ids[i] for i in pos}}, ["image"])[0]
                self.assertAlmostEqual(result["ndcg_at_10"], ndcg_score(truth, scores, k=10))
        # Give the oracle strictly descending scores for the project's deterministic tie order.
        tied = np.ones((1, 60))
        order = rank_scores(tied, ids)
        deterministic = np.empty_like(tied)
        np.put_along_axis(deterministic, order, np.arange(60, 0, -1)[None, :], axis=1)
        truth = np.zeros((1, 60)); truth[0, [2, 20, 21, 22, 23]] = 1
        result = evaluate(tied, ["q"], ids, {"q": {ids[i] for i in (2, 20, 21, 22, 23)}}, ["image"])[0]
        self.assertAlmostEqual(result["ndcg_at_10"], ndcg_score(truth, deterministic, k=10))

    def test_rrf_endpoints_and_validation(self):
        ids = ["c", "a", "b"]
        a = np.array([[0, 3, 2], [2, 2, 2]])
        b = np.array([[5, 1, 3], [1, 2, 3]])
        for weight, base in ((0, a), (1, b)):
            self.assertTrue(np.array_equal(rank_scores(weighted_rrf(a, b, ids, weight), ids), rank_scores(base, ids)))
        for weight in (float("nan"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                weighted_rrf(a, b, ids, weight)

    def test_rrf_reverse_must_recompute(self):
        ids = ["a", "b", "c"]
        a = np.array([[1, 2, 3], [5, 0, 6], [2, 3, 0]])
        b = np.array([[0, 4, 2], [1, 5, 3], [6, 2, 4]])
        self.assertFalse(np.allclose(weighted_rrf(a, b, ids, .5).T,
                                     weighted_rrf(a.T, b.T, ids, .5)))

    def test_paired_bootstrap_identical_zero(self):
        rows = self.rows()
        ci = bootstrap(rows, rows, repeats=20)
        self.assertTrue(ci["paired"])
        for value in ci["metrics"].values():
            self.assertEqual(value, {"estimate": 0, "low": 0, "high": 0})

    def test_grouping_repair_regression(self):
        good, bad = self.rows()[0], self.rows(11)[0]
        left, right = [], []
        # Five captions per image must travel together during resampling.
        for group in ("i0", "i1"):
            for j in range(5):
                a, b = (bad, good) if group == "i0" else (good, bad)
                left.append({**a, "query_id": f"{group}:{j}", "image_id": group})
                right.append({**b, "query_id": f"{group}:{j}", "image_id": group})
        # Positives must be the same even though ranks change.
        for a, b in zip(left, right):
            a["positive_ranks"] = {"positive": 11 if a["hit_recall_at_10"] == 0 else 1}
            b["positive_ranks"] = {"positive": 11 if b["hit_recall_at_10"] == 0 else 1}
        counts, cases = changes(left, list(reversed(right)))
        self.assertEqual(counts["repairs"], 5)
        self.assertEqual(counts["regressions"], 5)
        self.assertEqual(counts["net_recall_at_10"], 0)
        self.assertEqual(len(cases), 10)
        ci = bootstrap(left, right, repeats=200)
        self.assertEqual(ci["groups"], 2)
        self.assertEqual(ci["metrics"]["hit_recall_at_10"]["low"], -1)
        self.assertEqual(ci["metrics"]["hit_recall_at_10"]["high"], 1)
        with self.assertRaises(ValueError):
            changes(left, right[:-1])


if __name__ == "__main__":
    unittest.main()
