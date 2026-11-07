import unittest
from core import run, auc

class Tests(unittest.TestCase):

    def test_deterministic(self):
        self.assertEqual(run({'samples': 100}), run({'samples': 100}))

    def test_metrics(self):
        r = run({'samples': 100})
        self.assertEqual(r['train_rows'] + r['test_rows'], 100)
        self.assertTrue(0 <= r['roc_auc'] <= 1)
        self.assertEqual(r['scaler_fit_on'], 'training_only')

    def test_auc_ties(self):
        self.assertEqual(auc([0, 1], [0.5, 0.5]), 0.5)

    def test_small(self):
        with self.assertRaises(ValueError):
            run({'samples': 3})

    def test_single_class(self):
        with self.assertRaises(ValueError):
            run({'rows': [{'tenure': 1, 'tickets': 1, 'usage': 1, 'churn': 1}] * 40})
