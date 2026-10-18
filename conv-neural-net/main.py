"""Train and test the convolution-only network on local MNIST data."""
from src.mnist import load_mnist, verify_files
from src.model import ConvNeuralNet

# Edit these settings directly; no command-line arguments are needed.
TRAIN_SIZE = 500
TEST_SIZE = 200
EPOCHS = 3
NUM_LAYERS = 1
FILTERS = 4
SEED = 42


def main():
    verify_files()
    images, labels = load_mnist('train', limit=TRAIN_SIZE, seed=SEED)
    test_images, test_labels = load_mnist('test', limit=TEST_SIZE, seed=SEED)
    model = ConvNeuralNet(NUM_LAYERS, num_classes=10, filters=FILTERS, seed=SEED)
    print(f'MNIST: {len(images)} training images, {len(test_images)} test images', flush=True)
    for epoch in range(EPOCHS):
        loss = model.train((images, labels), epochs=1)[0]
        print(f'Epoch {epoch + 1}/{EPOCHS}: training loss {loss:.4f}', flush=True)
    correct = sum(model.predict(image) == label
                  for image, label in zip(test_images, test_labels))
    print(f'Held-out MNIST accuracy: {correct / len(test_labels):.1%} ({correct}/{len(test_labels)})')


if __name__ == '__main__':
    main()
