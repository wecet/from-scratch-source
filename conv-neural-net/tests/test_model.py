import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import ConvNeuralNet
def make_data(count, rng):
    labels = [rng.randrange(2) for _ in range(count)]
    images = [[[[rng.gauss(0, 0.05) + (int((y if label == 0 else x) % 2 == 0))
                 for x in range(4)] for y in range(4)]] for label in labels]
    return images, labels



def leaves(values, gradients):
    for i, value in enumerate(values):
        if isinstance(value, list):
            yield from leaves(value, gradients[i])
        else:
            yield values, i, gradients[i]


class ModelTests(unittest.TestCase):
    def test_gradients(self):
        images, labels = make_data(1, random.Random(9))
        for depth in (1, 2):
            model = ConvNeuralNet(depth, filters=2, seed=4)
            _, probabilities = model._loss(model._forward(images[0]), labels[0])
            gradients = model._backward(probabilities, labels[0])
            for (weight, bias), (dw, db) in zip(model.layers, gradients):
                for values, index, expected in leaves([weight, bias], [dw, db]):
                    original = values[index]
                    values[index] = original + 1e-6
                    plus = model._loss(model._forward(images[0]), labels[0])[0]
                    values[index] = original - 1e-6
                    minus = model._loss(model._forward(images[0]), labels[0])[0]
                    values[index] = original
                    self.assertAlmostEqual(expected, (plus - minus) / 2e-6, places=6)

    def test_learning(self):
        rng = random.Random(42)
        images, labels = make_data(24, rng)
        test_images, test_labels = make_data(16, rng)
        model = ConvNeuralNet(1, filters=4, seed=42)
        history = model.train((images, labels), 20)
        self.assertLess(history[-1], history[0] / 2)
        accuracy = sum(model.predict(x) == y for x, y in zip(test_images, test_labels)) / len(test_labels)
        self.assertGreater(accuracy, 0.9)
        self.assertAlmostEqual(sum(model.predict_proba(images[0])), 1)

    def test_validation(self):
        for depth in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                ConvNeuralNet(depth)
        model = ConvNeuralNet(1)
        for image in ([], [[[1], []]], [[[math.nan]]]):
            with self.assertRaises(ValueError):
                model.predict(image)
        loss, probabilities = model._loss([1000, -1000], 1)
        self.assertTrue(math.isfinite(loss))
        self.assertAlmostEqual(sum(probabilities), 1)


if __name__ == '__main__':
    unittest.main()
