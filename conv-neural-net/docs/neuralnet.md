# Understanding the convolutional neural network

This guide explains `src/model.py`, its mathematical model, and the small demo and tests. The implementation uses nested Python lists, loops, scalar arithmetic, `math`, and standard-library `random`. There is no NumPy, automatic differentiation, pooling, or dense layer.

## 1. What the model learns

An image classifier maps an image to a class index. Here, every convolution learns small spatial patterns through weights shared across image locations. Several convolutions can combine local patterns into larger ones. Training adjusts those weights to make the correct class more probable.

The computation is:

```text
image → convolution → ReLU → … → final convolution
      → center scores → softmax → class probabilities
```

ReLU appears only after hidden convolutions. With `num_layers=1`, there are no hidden layers: the model is a single linear convolution followed by softmax. It cannot learn nonlinear decision boundaries in that case.

The final convolution creates one feature map per class. The value at the center of each map is that class's score, also called a logit. This is a deliberately simple classifier readout, rather than a typical full-image CNN head.

## 2. Lists, dimensions, and parameters

An image has layout `image[channel][row][column]`, or `(C, H, W)` in mathematical notation. A grayscale image has one channel; RGB has three. The model processes one image at a time, so there is no batch dimension within an image.

Each layer is stored as `(weight, bias)` in `self.layers`:

- `weight[o][c][ky][kx]`: kernel coefficient from input channel `c` to output channel `o` at kernel row `ky`, column `kx`.
- `bias[o]`: one bias shared across all positions in output channel `o`.

Weights have dimensions `(C_out, C_in, 3, 3)`. A layer has `9 * C_out * C_in + C_out` trainable scalars. Hidden layers output `filters` channels; the last outputs `num_classes`. When there is only one layer, `filters` has no effect.

All images must have the configured number of channels, but separate images may have different heights and widths. Values should be normalized to a reasonable scale before training; the model does not perform normalization.

## 3. Convolution and padding

Let `A` denote a layer's input, `W` its weights, and `b` its biases. Before any activation, the output is

```text
Z[o,y,x] = b[o]
           + Σ_c Σ_ky=0..2 Σ_kx=0..2
             W[o,c,ky,kx] * A[c,y+ky-1,x+kx-1].
```

Input positions outside the image contribute zero. The implementation skips those positions instead of allocating a padded image. This is one-pixel zero padding. Stride is one: the kernel moves one pixel between outputs, preserving `H` and `W`.

Strictly, this operation is cross-correlation because it does not reverse the kernel. Neural network implementations conventionally call it convolution.

For one channel, at an interior pixel, a kernel with nine coefficients of `1` and bias `0` sums the surrounding 3×3 patch. A learned kernel can instead emphasize edges, differences, or other patterns. Each output channel learns a different collection of kernels across the input channels.

Weight sharing means the same coefficient contributes at many positions. This reduces the parameter count and causes gradients for that coefficient to accumulate from all those positions.

## 4. Hidden activations and class scores

Hidden layers apply the rectified linear unit:

```text
A[o,y,x] = ReLU(Z[o,y,x]) = max(0, Z[o,y,x]).
```

ReLU introduces nonlinearity. Without an activation between convolutions, stacking linear operations would still produce a linear mapping from input to scores.

The final layer leaves its outputs unrestricted: negative logits are useful and valid. For class `j`, its logit is

```text
z[j] = Z_final[j, H // 2, W // 2].
```

For even dimensions, integer division selects the lower/right member of the central pair. There is no averaging or maximum over spatial positions, so this is not pooling.

After `L` stride-one 3×3 convolutions, the theoretical receptive field has side length `1 + 2L`. Only the image region within that distance of the selected center can influence the prediction. Padding may occupy some of that region. A shallow model can therefore ignore most of a large image. Convolutions share weights across space, but this particular center readout does not provide translation-invariant image classification.

## 5. Softmax and cross-entropy

Softmax converts `K` logits into probabilities:

```text
p[j] = exp(z[j]) / Σ_k exp(z[k]).
```

Probabilities are nonnegative and sum to one. For the correct class index `t`, cross-entropy for one image is

```text
loss = -log(p[t]).
```

Increasing the correct class's probability lowers this loss. Softmax probabilities are model outputs, not a guarantee of calibrated confidence.

Directly exponentiating large logits can overflow. `_loss` subtracts their maximum, `m = max(z)`, without changing the probabilities:

```text
s[j] = z[j] - m
S = Σ_j exp(s[j])
p[j] = exp(s[j]) / S
loss = log(S) - s[t].
```

For example, logits `[1000, -1000]` become `[0, -2000]`. The incorrect-class loss can be computed as approximately `2000` even if that class's probability underflows to zero. The code never needs to take `log(0)`.

This improves numerical stability for finite logits; it does not prevent excessively large training updates from eventually creating nonfinite values.

## 6. Backpropagation, step by step

Backpropagation uses the chain rule to calculate how each parameter affects the loss. Every derivative below is for one image, matching the online training loop.

### Logits and the center readout

Combining softmax and cross-entropy gives a simple derivative:

```text
dloss/dz[j] = p[j] - indicator(j == t).
```

`_backward` copies the probability list and subtracts `1` at the label index. For example, probabilities `[0.2, 0.8]` with label `1` give logit gradients `[0.2, -0.2]`.

Because only the center locations supplied the logits, the gradient with respect to the final feature maps starts at zero everywhere except those locations. The variable `dx` stores these incoming activation gradients.

### ReLU derivative

At a hidden layer, let `G` be the incoming derivative with respect to its output activation. The derivative with respect to its preactivation is

```text
D[o,y,x] = G[o,y,x]  if Z[o,y,x] > 0
           0         otherwise.
```

The code checks the cached output rather than storing `Z`: a positive ReLU output implies a positive preactivation. At exactly zero, it chooses derivative zero, a conventional choice at ReLU's nondifferentiable point. The final convolution has no ReLU, so its gradient passes through directly.

### Convolution derivatives

For each valid input location `iy = y + ky - 1`, `ix = x + kx - 1`, the scalar forward contribution is

```text
Z[o,y,x] += W[o,c,ky,kx] * A[c,iy,ix].
```

Its backward contributions are

```text
dW[o,c,ky,kx] += D[o,y,x] * A[c,iy,ix]
dA[c,iy,ix]  += D[o,y,x] * W[o,c,ky,kx]
db[o]       += D[o,y,x].
```

The bias accumulation happens once per output position, outside the kernel loops. The weight and input accumulations happen for each valid kernel contribution. Repeated contributions must be added, because one parameter or input value participates in multiple forward operations.

`previous` stores `dA`, which becomes the incoming gradient for the preceding layer. Layers are visited in reverse order. Their parameter gradients are collected in reverse order, then reversed back to match `self.layers`.

No parameters change during this calculation. Updating an early-visited layer immediately would make later derivatives use different weights from the forward pass.

## 7. Updating parameters without a configurable learning rate

The constructor no longer accepts a learning-rate argument. `_step` uses a fixed scalar step of `0.05`:

```text
parameter_new = parameter_old - 0.05 * derivative.
```

For example, a coefficient of `0.3` with derivative `0.4` becomes `0.28`.

The minus sign moves in the direction of decreasing loss locally. This first-order direction does not guarantee that a finite step decreases the loss on every image.

A gradient update always has an effective step size. Subtracting the unscaled gradient would implicitly choose a step size of `1`; it would not remove the concept mathematically. Keeping a fixed `0.05` removes the configuration from the API while retaining a modest step for this small demonstration. It is still mathematically a learning rate, and it is not guaranteed to suit arbitrary datasets or deeper models.

Training uses online stochastic gradient descent: update after each image. There are no mini-batches, momentum buffers, weight decay, or adaptive optimizer state.

## 8. Every function in `src/model.py`

### `zeros(channels, height, width)`

Builds a three-dimensional nested list filled with `0.0`. Each nested list is created independently so writing to one pixel does not accidentally change another row or channel. Used for activations and gradient buffers. A list multiplication shortcut for nested mutable lists could create shared references.

### `ConvNeuralNet.__init__(...)`

Validates positive integer layer, class, channel, and filter counts; rejects booleans as counts; requires at least two classes. Stores the architecture and creates a private `random.Random(seed)` generator before building weights. A private generator avoids changing global random state. The seed makes initialization and subsequent shuffling reproducible for the same sequence of calls.

Arguments are `num_layers`, `num_classes=2`, `in_channels=1`, `filters=8`, and `seed=0`.

### `_build_model()`

Creates the layer list, calling `_add_layer` once per convolution. The first layer reads `in_channels`; subsequent layers read `filters`. This works because every nonfinal layer outputs `filters` channels. Calling this private method again would discard learned weights and initialize a fresh model.

### `_add_layer(in_channels)`

Chooses the output channel count based on whether this is the last layer, builds the 3×3 kernels, and initializes biases to zero. Each weight is sampled from a zero-mean Gaussian with standard deviation

```text
sqrt(2 / (9 * in_channels)).
```

This is He-style initialization based on the number of incoming kernel coefficients. It is intended to keep hidden ReLU activation scales reasonable. The same simple rule is used for the final linear convolution, although that layer has no ReLU. Random weights also break the symmetry that would occur if all filters started identically.

### `_validate_image(image)`

Checks channel count, positive dimensions, rectangular rows and channels, and finite numeric values. Invalid image structures raise `ValueError`. This prevents later convolution indexing from failing on ragged or empty input. It does not rescale pixels or copy the input.

### `_forward(image)`

Computes every convolution and hidden ReLU, caches the input and each layer's output in `self.activations`, then returns the center logits. Caching makes the original values available to backpropagation. This private method assumes input validation has already happened.

The cache is replaced on every forward call, including prediction. Therefore `_backward` must immediately follow the forward pass whose derivatives it needs; the model instance is not designed for concurrent training and prediction.

### `_loss(logits, label)`

A static method returning `(loss, probabilities)` using stable softmax cross-entropy. It needs no model state. The caller supplies a valid label; training validates labels beforehand. Keeping probabilities with the loss avoids recalculating softmax for backward propagation.

### `_backward(probabilities, label)`

Uses the latest activation cache to compute a `(dweight, dbias)` pair per layer. It seeds the center gradients, applies hidden ReLU masks, and accumulates convolution derivatives. Returns gradients in forward layer order without changing parameters.

### `_step(gradients)`

Traverses every bias and weight and subtracts `0.05` times its derivative. The gradient structure mirrors `self.layers`, allowing explicit loops with no dynamic attribute checks.

### `train(training_data, epochs, shuffle=True)`

Accepts `(images, labels)`, validates the full dataset and epoch count before updates, and runs the requested number of passes through the data. Each image goes through forward, loss, backward, and update. Optional shuffling changes sample order each epoch without rearranging the original dataset.

Returns a list containing the mean observed loss for each epoch. These are losses measured before each sample's update, across a changing model; they are not a fresh dataset evaluation using only the final weights of that epoch. Later calls continue training existing parameters and continue the random generator's state.

### `predict_proba(image)`

Validates one image and returns a list of class probabilities. It reuses `_loss` with dummy label `0` to obtain softmax probabilities; the computed loss is discarded. The dummy label has no effect on the returned probabilities. Prediction changes the activation cache but not the weights.

### `predict(image)`

Returns the integer index with the largest probability. If probabilities tie, Python's `max` returns the first such index. It accepts one image, not a dataset; use a list comprehension to predict several images.

## 9. Demo and tests

`main.py` runs MNIST by default. Its top-level constants select training/test subset sizes, epochs, layer count, hidden filter count, and seed. `main()` verifies local archives, loads separate train and test subsets, trains a ten-class model one epoch at a time, prints each epoch's loss, and reports held-out accuracy. There are no command-line arguments. The entry-point guard runs training only when the file is executed directly.

The synthetic `make_data(count, rng)` helper lives in `tests/test_model.py` solely as a small test fixture. It creates noisy horizontal/vertical stripes for quick gradient and learning checks.

In `tests/test_model.py`, `leaves` recursively visits scalar parameters alongside their analytic gradients. `test_gradients` checks every weight and bias in one- and two-layer models against central finite differences:

```text
dloss/dparameter ≈ (loss(parameter + epsilon)
                   - loss(parameter - epsilon)) / (2 * epsilon),
epsilon = 0.000001.
```

This checks the full chain from parameters through center scores and loss. Numerical derivatives can be unreliable at a ReLU kink, so the deterministic test case uses a small tolerance and fixed initialization. `test_learning` checks decreasing loss and held-out accuracy on the synthetic task. `test_validation` checks invalid inputs and a large-logit loss example.

Run from the repository root:

```sh
python3 conv-neural-net/main.py
python3 -m unittest discover -s conv-neural-net/tests -v
```

## 10. Practical limits

A forward convolution takes roughly `H * W * C_out * C_in * 9` multiply-add contributions per image. Backpropagation has a similar order of work. Python loops and nested-list storage make this suitable for small educational examples, not large-scale training.

Activations require storage proportional to the total number of pixels across layers; Python objects add considerable overhead. The code computes full feature maps even where positions cannot influence the selected center score, favoring uniform loops over optimization.

There is no pooling, full-image classification head, model persistence, automatic normalization, GPU support, or configurable optimizer. The fixed update size and initialization are demonstration choices. The most important architectural limit is that only the center receptive field affects the prediction.

## 11. Loading and testing on MNIST

`src/mnist.py` adds dataset handling without changing the model. MNIST has 60,000 training and 10,000 test examples, ten classes (digits 0–9), and 28×28 single-channel byte images.

`verify_files(directory)` checks the MD5 digest of each compressed archive against the expected MNIST digest. These checks detect incorrect or damaged local files.

`load_mnist(split, limit, seed, directory)` decompresses the selected image and label archives using `gzip`. `struct.unpack` reads unsigned big-endian IDX headers: image magic number 2051 followed by count, height, and width; label magic number 2049 followed by count. The loader checks magic numbers, expected dimensions and split counts, payload lengths, and digit label ranges.

A private seeded random generator selects indices without replacement within the requested split. For each selected image, the loader constructs a `[1][28][28]` list and divides byte values by 255. Labels become Python integers. It returns `(images, labels)` directly usable by `train`. `limit=None` selects the entire split, but turning all images into nested Python float lists requires much more memory than their original byte representation. Even when loading a subset, the compressed split is decompressed into a byte buffer first.

`main.py` uses the settings described above to run the MNIST experiment. Its held-out accuracy is the number of correctly classified test images divided by the number of test images. It prints results to the terminal without writing a report file.

The test subset is held out from gradient updates. The seed controls both subset selection and model randomness. These small seeded subsets provide a reproducible practical check, not a full-dataset benchmark. Do not infer broad MNIST performance from a small sample, or treat the test set as a tuning set.

The existing model is deliberately unchanged: a single convolution reads only a central 3×3 patch of each 28×28 image. Most of each digit is invisible to this classifier. Extra convolution layers expand the receptive field but also increase runtime; the experiment does not silently add pooling, flattening, or a dense classifier.

`tests/test_mnist.py` checks archive integrity, shapes, normalization, labels, deterministic sampling, and invalid loader arguments. These integration tests skip when dataset files are absent so the original model tests can run without a download.
