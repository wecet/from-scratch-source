"""Convolutional classifier using lists, scalar arithmetic, math, and random."""
import math
import random


def zeros(channels, height, width):
    return [[[0.0 for _ in range(width)] for _ in range(height)] for _ in range(channels)]


class ConvNeuralNet:
    def __init__(self, num_layers: int, num_classes=2, in_channels=1,
                 filters=8, seed=0):
        for name, value in (("num_layers", num_layers), ("num_classes", num_classes),
                            ("in_channels", in_channels), ("filters", filters)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_layers, self.num_classes = num_layers, num_classes
        self.in_channels, self.filters = in_channels, filters
        self.rng = random.Random(seed)
        self._build_model()

    def _build_model(self):
        self.layers = []
        for i in range(self.num_layers):
            self._add_layer(self.in_channels if i == 0 else self.filters)

    def _add_layer(self, in_channels):
        # The last convolution produces one score map per class.
        out_channels = self.num_classes if len(self.layers) == self.num_layers - 1 else self.filters
        scale = math.sqrt(2 / (9 * in_channels))
        weight = [[[[self.rng.gauss(0, scale) for _ in range(3)] for _ in range(3)]
                   for _ in range(in_channels)] for _ in range(out_channels)]
        self.layers.append((weight, [0.0] * out_channels))

    def _validate_image(self, image):
        try:
            height, width = len(image[0]), len(image[0][0])
            valid = (len(image) == self.in_channels and height > 0 and width > 0
                     and all(len(channel) == height for channel in image)
                     and all(len(row) == width for channel in image for row in channel)
                     and all(math.isfinite(value) for channel in image for row in channel for value in row))
        except (IndexError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("image must be a finite rectangular (channels, height, width) list")

    def _forward(self, image):
        """Forward one image; cache activations for manual backpropagation."""
        self.activations = [image]
        height, width = len(image[0]), len(image[0][0])
        for index, (weight, bias) in enumerate(self.layers):
            output = zeros(len(weight), height, width)
            for o in range(len(weight)):
                for y in range(height):
                    for x in range(width):
                        value = bias[o]
                        for c in range(len(image)):
                            for ky in range(3):
                                iy = y + ky - 1
                                if not 0 <= iy < height:
                                    continue
                                for kx in range(3):
                                    ix = x + kx - 1
                                    if 0 <= ix < width:
                                        value += image[c][iy][ix] * weight[o][c][ky][kx]
                        output[o][y][x] = value if index == self.num_layers - 1 else max(0.0, value)
            image = output
            self.activations.append(image)
        # Read a single spatial location: no pooling or dense classifier.
        return [channel[height // 2][width // 2] for channel in image]

    @staticmethod
    def _loss(logits, label):
        maximum = max(logits)
        shifted = [value - maximum for value in logits]
        exps = [math.exp(value) for value in shifted]
        total = sum(exps)
        probabilities = [value / total for value in exps]
        return math.log(total) - shifted[label], probabilities

    def _backward(self, probabilities, label):
        """Return all parameter gradients before any weights are updated."""
        grad = probabilities[:]
        grad[label] -= 1
        height, width = len(self.activations[0][0]), len(self.activations[0][0][0])
        dx = zeros(self.num_classes, height, width)
        for c, value in enumerate(grad):
            dx[c][height // 2][width // 2] = value
        conv_gradients = []
        for index in reversed(range(self.num_layers)):
            image, output = self.activations[index:index+2]
            weight, _ = self.layers[index]
            dweight = [zeros(len(image), 3, 3) for _ in range(len(weight))]
            dbias = [0.0] * len(weight)
            previous = zeros(len(image), height, width)
            for o in range(len(weight)):
                for y in range(height):
                    for x in range(width):
                        g = dx[o][y][x] if index == self.num_layers - 1 or output[o][y][x] > 0 else 0.0
                        dbias[o] += g
                        for c in range(len(image)):
                            for ky in range(3):
                                iy = y + ky - 1
                                if not 0 <= iy < height:
                                    continue
                                for kx in range(3):
                                    ix = x + kx - 1
                                    if 0 <= ix < width:
                                        dweight[o][c][ky][kx] += g * image[c][iy][ix]
                                        previous[c][iy][ix] += g * weight[o][c][ky][kx]
            conv_gradients.append((dweight, dbias))
            dx = previous
        return list(reversed(conv_gradients))

    def _step(self, gradients):
        # Fixed gradient step keeps the public API small.
        rate = 0.05
        for (weight, bias), (dweight, dbias) in zip(self.layers, gradients):
            for o in range(len(weight)):
                bias[o] -= rate * dbias[o]
                for c in range(len(weight[o])):
                    for y in range(3):
                        for x in range(3):
                            weight[o][c][y][x] -= rate * dweight[o][c][y][x]

    def train(self, training_data, epochs: int, shuffle=True):
        """Online SGD on (list_of_CHW_images, integer_labels); return epoch losses."""
        images, labels = training_data
        if not images or len(images) != len(labels):
            raise ValueError("provide a nonempty image list with one label per image")
        for image in images:
            self._validate_image(image)
        if any(type(label) is not int or not 0 <= label < self.num_classes for label in labels):
            raise ValueError("labels must be integers between 0 and num_classes - 1")
        if type(epochs) is not int or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        history = []
        order = list(range(len(images)))
        for _ in range(epochs):
            if shuffle:
                self.rng.shuffle(order)
            total = 0.0
            for i in order:
                loss, probabilities = self._loss(self._forward(images[i]), labels[i])
                self._step(self._backward(probabilities, labels[i]))
                total += loss
            history.append(total / len(images))
        return history

    def predict_proba(self, image):
        """Return class probabilities for one (channels, height, width) image."""
        self._validate_image(image)
        return self._loss(self._forward(image), 0)[1]

    def predict(self, image):
        probabilities = self.predict_proba(image)
        return max(range(self.num_classes), key=probabilities.__getitem__)
