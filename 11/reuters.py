# Import necessary libraries
import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# Constants for dataset and training
MAX_WORDS = 10000
BATCH_SIZE = 512


class NewsClassifier:
    def __init__(self, max_words=MAX_WORDS):
        self.max_words = max_words
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self.model = None
        print("[INFO] NewsClassifier initialized.")

    def prepare_data(self):
        print("[INFO] Preparing data...")
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = reuters.load_data(num_words=self.max_words)

        # Convert to one-hot encoding
        self.x_train = self._one_hot_encode(x_train_raw, self.max_words)
        self.y_train = np.array(y_train_raw, dtype="int32")

        self.x_test = self._one_hot_encode(x_test_raw, self.max_words)
        self.y_test = np.array(y_test_raw, dtype="int32")
        print("[INFO] Data prepared.")

    def build_model(self):
        print("[INFO] Building model...")
        self.model = models.Sequential()
        self.model.add(layers.Dense(64, activation="relu", input_shape=(self.max_words,)))
        self.model.add(layers.Dense(64, activation="relu"))
        self.model.add(layers.Dense(46, activation="softmax"))

        # Compile the model
        self.model.compile(optimizer="rmsprop",
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])
        print("[INFO] Model built.")

    def train(self, epochs=10):
        if self.model is None:
            print("[WARNING] Model is not defined. Call `build_model()` first.")
            return

        print(f"[INFO] Training model for {epochs} epochs...")
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            validation_split=0.2,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        print("[INFO] Model training completed.")

    def plot_loss(self):
        if not hasattr(self, 'history'):
            print("[WARNING] No training history found. Train the model first.")
            return

        # Plot training and validation loss
        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss, "r", label="Training Loss")
        plt.plot(epochs, val_loss, "r--", label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        if not hasattr(self, 'history'):
            print("[WARNING] No training history found. Train the model first.")
            return

        # Plot training and validation accuracy
        accuracy = self.history.history["accuracy"]
        val_accuracy = self.history.history["val_accuracy"]
        epochs = range(1, len(accuracy) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, accuracy, "b", label="Training Accuracy")
        plt.plot(epochs, val_accuracy, "b--", label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def evaluate(self):
        if self.model is None:
            print("[WARNING] Model is not defined. Call `build_model()` first.")
            return

        print("[INFO] Evaluating model...")
        results = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        print(f"[INFO] Test Loss: {results[0]:.4f}")
        print(f"[INFO] Test Accuracy: {results[1]:.4f}")

    def _one_hot_encode(self, sequences, dimension):
        # Convert sequences to binary matrix
        encoded = np.zeros((len(sequences), dimension), dtype="float32")
        for i, sequence in enumerate(sequences):
            encoded[i, sequence] = 1.0
        return encoded


# Main script
if __name__ == "__main__":
    print("[INFO] Starting NewsClassifier...")

    # Instantiate the classifier
    classifier = NewsClassifier(max_words=MAX_WORDS)

    # Step-by-step process
    classifier.prepare_data()   # Prepare the data
    classifier.build_model()   # Build the model
    classifier.train(epochs=5) # Train the model (reduced epochs to 5 for quick testing)
    classifier.plot_loss()     # Plot training and validation loss
    classifier.plot_accuracy() # Plot training and validation accuracy
    classifier.evaluate()      # Evaluate the model on the test dataset
