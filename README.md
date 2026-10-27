# From scratch

Small machine learning projects built from first principles with Python lists and scalar math. No third-party dependencies are required.

## Projects

- [Convolutional neural network](conv-neural-net/README.md): convolution, manual backpropagation, and stochastic gradient descent, trained on MNIST.

More projects are planned.

## Getting started

Requires Python 3.10 or newer.

```sh
git clone https://github.com/wecet/from-scratch-source.git
cd from-scratch-source
python3 -m unittest discover -s conv-neural-net/tests -v
```

The model tests need no data. Datasets are not included in the repository. To run the MNIST experiment, first download the dataset as described in [the CNN README](conv-neural-net/README.md#mnist), then run:

```sh
python3 conv-neural-net/main.py
```

You can also run `uv sync` to create a virtual environment from `pyproject.toml`, then run the same commands with `uv run python`.

## License

The code is released under the [MIT License](LICENSE). The MNIST dataset is not part of this repository and is not covered by this license.
