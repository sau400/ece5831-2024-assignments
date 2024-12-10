# Import necessary libraries
import os
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# Suppress TensorFlow logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants for dataset and training
MAX_FEATURES = 10000
BATCH_SIZE = 512


class MovieReviews:
    def __init__(self, max_features=MAX_FEATURES):
        self.max_features = max_features
        self.x_train, self.y_train, self.x_test, self.y_test = self._prepare_data()
        self.model = self._build_model()

    def _prepare_data(self):
        # Load IMDB dataset
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = imdb.load_data(num_words=self.max_features)

        # Vectorize input data
        x_train = self._one_hot_encode(x_train_raw)
        y_train = np.array(y_train_raw, dtype="float32")

        x_test = self._one_hot_encode(x_test_raw)
        y_test = np.array(y_test_raw, dtype="float32")

        return x_train, y_train, x_test, y_test

    def _one_hot_encode(self, sequences):
        # Convert sequences into binary feature vectors
        encoded = np.zeros((len(sequences), self.max_features), dtype="float32")
        for i, sequence in enumerate(sequences):
            encoded[i, sequence] = 1.0
        return encoded

    def _build_model(self):
        # Create the neural network
        model = models.Sequential()
        model.add(layers.Dense(32, activation="relu", input_shape=(self.max_features,)))
        model.add(layers.Dense(16, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def train(self, epochs=10):
        # Train the model and return history
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            verbose=2
        )
        return history

    def evaluate(self):
        # Evaluate the model on test data
        results = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        print(f"[INFO] Test Loss: {results[0]:.4f}")
        print(f"[INFO] Test Accuracy: {results[1]:.4f}")

    def plot_metrics(self, history, metric_name, plot_title, ylabel):
        # Plot training and validation metrics
        history_data = history.history
        train_metric = history_data[metric_name]
        val_metric = history_data[f"val_{metric_name}"]
        epochs = range(1, len(train_metric) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_metric, "b", label=f"Training {ylabel}")
        plt.plot(epochs, val_metric, "b--", label=f"Validation {ylabel}")
        plt.title(plot_title)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()


# Main script
if __name__ == "__main__":
    # Initialize the class
    reviews = MovieReviews(max_features=MAX_FEATURES)

    # Train the model
    print("Training the model...")
    history = reviews.train(epochs=10)

    # Evaluate the model
    print("Evaluating the model on test data...")
    reviews.evaluate()

    # Plot training and validation loss
    reviews.plot_metrics(history, metric_name="loss", plot_title="Training vs Validation Loss", ylabel="Loss")

    # Plot training and validation accuracy
    reviews.plot_metrics(history, metric_name="accuracy", plot_title="Training vs Validation Accuracy", ylabel="Accuracy")
