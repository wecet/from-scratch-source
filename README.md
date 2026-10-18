# From scratch

Small machine learning projects built from first principles with Python lists and scalar math. No third-party dependencies are required.

- [Convolutional neural network](conv-neural-net/README.md): convolution, manual backpropagation, and stochastic gradient descent.
- `vlm/`: reserved for future work.

Requires Python 3.10 or newer. From the repository root:

```sh
python3 conv-neural-net/main.py
python3 -m unittest discover -s conv-neural-net/tests -v
```

Optionally use `uv sync` to create a virtual environment from `pyproject.toml`, then run the same commands with `uv run python`.
