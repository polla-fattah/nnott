from data_utils import DataUtility
from network import Network
from trainer import Trainer


def main():

    # 1. Load the data
    data_util = DataUtility(data_dir="data")
    X_train, y_train, X_test, y_test = data_util.load_data()

    print("Train images:", X_train.shape)   # (60000, 28, 28)
    print("Train labels:", y_train.shape)   # (60000,)
    print("Test images:", X_test.shape)     # (10000, 28, 28)
    print("Test labels:", y_test.shape)     # (10000,)

    # 1b. Show some of the training data visually
    DataUtility.show_samples(X_train, y_train, num_samples=10, title="Training samples")

    # 2. Create network & trainer
    # for MNIST-like data: 28x28, 10 classes
    num_classes = 10  # labels 0–9

    network = Network(
        input_size=28 * 28,       # explicit, matches (28,28)
        num_classes=num_classes,
        hidden_sizes=(128, 64),
        learning_rate=0.01,
        activation='relu' #sigmoid
    )

    trainer = Trainer(network, num_classes=num_classes)

    # 2. Start training (tip: start with 1–2 epochs to make it faster)
    trainer.train(X_train, y_train, epochs=3, verbose=True)

    # 3. Test the result (evaluate)
    trainer.evaluate(X_test, y_test)

    # 3b. Show 10 random test images with true/pred labels
    trainer.show_random_predictions(X_test, y_test, num_samples=10)


if __name__ == "__main__":
    main()
