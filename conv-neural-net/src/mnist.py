"""Read the original compressed MNIST IDX files using only the standard library."""
import gzip
import hashlib
from pathlib import Path
import random
import struct


RESOURCE_DIR = Path(__file__).resolve().parents[1] / 'resources' / 'mnist'
BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
# Published MNIST file digests; used to catch incomplete or incorrect downloads.
FILES = {
    'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
}


def verify_files(directory=RESOURCE_DIR):
    """Validate all four compressed archives before using the dataset."""
    directory = Path(directory)
    for filename, expected in FILES.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f'Missing MNIST file: {path}; download from {BASE_URL}{filename}')
        if hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'MNIST checksum mismatch: {path}')


def load_mnist(split='train', limit=500, seed=42, directory=RESOURCE_DIR):
    """Load a seeded random subset as normalized CHW lists and integer labels.

    Use limit=None for the full split (large memory cost with Python floats).
    Sampling uses only the requested split and never mixes train and test data.
    """
    if split not in ('train', 'test'):
        raise ValueError("split must be 'train' or 'test'")
    prefix = 'train' if split == 'train' else 't10k'
    directory = Path(directory)
    with gzip.open(directory / f'{prefix}-images-idx3-ubyte.gz', 'rb') as source:
        images = source.read()
    with gzip.open(directory / f'{prefix}-labels-idx1-ubyte.gz', 'rb') as source:
        labels = source.read()
    if len(images) < 16 or len(labels) < 8:
        raise ValueError('Truncated MNIST IDX headers')
    image_magic, count, height, width = struct.unpack('>IIII', images[:16])
    label_magic, label_count = struct.unpack('>II', labels[:8])
    if (image_magic != 2051 or label_magic != 2049 or count != label_count
            or count != (60000 if split == 'train' else 10000)
            or (height, width) != (28, 28)):
        raise ValueError('Invalid MNIST IDX headers')
    if len(images) != 16 + count * height * width or len(labels) != 8 + count:
        raise ValueError('MNIST payload size does not match its headers')
    if any(label > 9 for label in labels[8:]):
        raise ValueError('MNIST labels must be digits 0 through 9')
    if limit is None:
        limit = count
    if type(limit) is not int or not 1 <= limit <= count:
        raise ValueError(f'limit must be between 1 and {count}, or None')
    indices = random.Random(seed).sample(range(count), limit)
    result, targets = [], []
    for index in indices:
        offset = 16 + index * height * width
        channel = [[images[offset + row * width + col] / 255.0
                    for col in range(width)] for row in range(height)]
        result.append([channel])
        targets.append(labels[8 + index])
    return result, targets
