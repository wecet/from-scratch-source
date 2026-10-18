"""Integration checks; skipped when the local MNIST archives are absent."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mnist import FILES, RESOURCE_DIR, load_mnist, verify_files


@unittest.skipUnless(all((RESOURCE_DIR / name).is_file() for name in FILES),
                     'MNIST archives have not been downloaded')
class MnistTests(unittest.TestCase):
    def test_integrity_and_normalization(self):
        verify_files()
        for split in ('train', 'test'):
            images, labels = load_mnist(split, limit=3)
            self.assertEqual(len(images), len(labels))
            for image, label in zip(images, labels):
                self.assertEqual(len(image), 1)
                self.assertEqual(len(image[0]), 28)
                self.assertTrue(all(len(row) == 28 for row in image[0]))
                self.assertTrue(all(math.isfinite(value) and 0 <= value <= 1
                                    for row in image[0] for value in row))
                self.assertIn(label, range(10))

    def test_sampling_and_arguments(self):
        self.assertEqual(load_mnist(limit=2), load_mnist(limit=2))
        for limit in (0, -1, True, 60001):
            with self.assertRaises(ValueError):
                load_mnist(limit=limit)
        with self.assertRaises(ValueError):
            load_mnist('validation')
