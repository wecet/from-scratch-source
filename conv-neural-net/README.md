# A simple convolutional neural network

A small classifier built with Python lists, loops, scalar arithmetic, and `math`. Standard-library `random` provides seeded initialization, shuffling, and synthetic test fixtures. There is no NumPy or external library dependency.

## Architecture

`num_layers` counts 3×3 convolutions with stride 1 and zero padding, preserving image height and width. Hidden convolutions have `filters` output channels followed by ReLU. The final convolution has `num_classes` output channels with no ReLU. With one layer, the network is a single convolutional classifier.

For simplicity, class scores come from the center pixel of the final output channels (index `height // 2, width // 2`). Softmax converts those scores to probabilities. There is no pooling or dense layer. This readout only sees the center's receptive field: at most `1 + 2 * num_layers` pixels along each dimension, so shallow networks will ignore distant features.

Training implements stable softmax cross-entropy, manual backpropagation, and online stochastic gradient descent: one image per weight update. All gradients are computed before updating weights. The gradient step is fixed at 0.05; there is no learning-rate constructor argument.

## Run

From the repository root:

```sh
python3 conv-neural-net/main.py
python3 -m unittest discover -s conv-neural-net/tests -v
```

The default program trains on MNIST and reports epoch losses and held-out digit classification accuracy. It reads the local dataset and downloads nothing. Tests check numerical gradients for one- and two-layer models, learning, input validation, and stable loss calculations.

## Use

From `conv-neural-net/`:

```python
from src.model import ConvNeuralNet

# One grayscale 3×3 image. Each image is [channels][rows][columns].
image = [[[1.0, 0.0, 1.0],
          [1.0, 0.0, 1.0],
          [1.0, 0.0, 1.0]]]
model = ConvNeuralNet(num_layers=2, filters=4, num_classes=2, seed=0)
losses = model.train(([image], [1]), epochs=10)
probabilities = model.predict_proba(image)  # list of class probabilities
class_id = model.predict(image)            # integer class index
```

Provide a nonempty list of images and one integer label per image in `[0, num_classes)`. Images must be finite, rectangular nested lists with `in_channels` channels. Normalize byte-valued pixels yourself, for example by dividing each value by 255. Images can have different spatial sizes because they are processed individually.

`train` returns one mean loss per epoch and continues training existing weights on subsequent calls. `shuffle=False` preserves input order. Prediction accepts one image at a time; use `[model.predict(image) for image in images]` for multiple images.

All model calculations live in `src/model.py`; `main.py` contains the demo. This is a small learning implementation: Python loops are slow for large images, and model saving and advanced optimizers are not included.

See [the implementation and mathematics guide](docs/neuralnet.md) for a function-by-function explanation, worked examples, and backpropagation equations.

## MNIST

The full original dataset is stored locally in `resources/mnist/` as four compressed IDX archives: 60,000 training images/labels and 10,000 test images/labels. Each image is 28×28 grayscale. Resource files are excluded from Git.

Run the MNIST experiment from the repository root:

```sh
python3 conv-neural-net/main.py
```

Edit `TRAIN_SIZE`, `TEST_SIZE`, `EPOCHS`, `NUM_LAYERS`, `FILTERS`, and `SEED` at the top of `main.py` to change the experiment. No command-line arguments or argument parser are used.

The defaults sample 500 training images and 200 images from the separate official test split, using seed 42, then train a single convolution for three epochs. Images retain their original 28×28 dimensions, with pixels divided by 255. Test data is never used for updates. All four local archives are checksum-verified before loading.

Pure Python convolution can take substantial time, especially with multiple layers. The default one-layer model only sees the central 3×3 patch, so its accuracy should not be compared with a conventional full-image MNIST CNN. The default run achieved **44.0% test accuracy**, up from 12.0% before training; the training-majority baseline scored 8.0%. Training loss fell from 2.1428 to 1.8427. This run took about 48 seconds on the local machine.
