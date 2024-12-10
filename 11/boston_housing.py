# Suppress TensorFlow informational logs and optional oneDNN optimizations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress logs below warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations if needed

# Import required packages
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

# Define the Boston_Housing class
class BostonHousing:
    def __init__(self, num_epochs=20, batch_size=16):
        """Initialize the BostonHousing class."""
        self.model = None
        self.all_mae_histories = []
        self.epochs = num_epochs
        self.batch_size = batch_size
        self._load_and_normalize_data()

    def _load_and_normalize_data(self):
        """Load and normalize the Boston Housing dataset."""
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        self.x_train = (x_train - mean) / std
        self.x_test = (x_test - mean) / std
        self.y_train = y_train
        self.y_test = y_test

    def _build_model(self):
        """Build and compile the neural network model."""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.x_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        return model

    def k_fold_validations(self, k=4):
        """Perform k-fold cross-validation."""
        num_val_samples = len(self.x_train) // k
        for i in range(k):
            print(f"Processing fold #{i + 1}")
            val_data = self.x_train[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = self.y_train[i * num_val_samples: (i + 1) * num_val_samples]
            partial_train_data = np.concatenate(
                [self.x_train[:i * num_val_samples], self.x_train[(i + 1) * num_val_samples:]],
                axis=0
            )
            partial_train_targets = np.concatenate(
                [self.y_train[:i * num_val_samples], self.y_train[(i + 1) * num_val_samples:]],
                axis=0
            )

            # Build and train the model for this fold
            model = self._build_model()
            history = model.fit(
                partial_train_data, partial_train_targets,
                validation_data=(val_data, val_targets),
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0
            )
            self.all_mae_histories.append(history.history['val_mae'])

    def plot_validation_mae(self):
        """Plot the validation mean absolute error (MAE)."""
        average_mae_history = [
            np.mean([x[i] for x in self.all_mae_histories]) for i in range(self.epochs)
        ]
        plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
        plt.title("Validation MAE Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Validation MAE")
        plt.show()

    def train(self, num_epochs):
        """Train the model on the full training dataset."""
        if self.model is None:
            self.model = self._build_model()
        self.model.fit(
            self.x_train, self.y_train,
            epochs=num_epochs,
            batch_size=self.batch_size,
            verbose=0
        )

    def evaluate(self):
        """Evaluate the trained model on the test dataset."""
        if self.model is None:
            print("[INFO] Model is not trained yet.")
            return
        test_loss, test_mae = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"[INFO] Test loss: {test_loss}")
        print(f"[INFO] Test MAE: {test_mae}")


# Main block
if __name__ == "__main__":
    # Initialize the class
    boston_housing = BostonHousing(num_epochs=20, batch_size=16)
    
    # Perform k-fold validation
    boston_housing.k_fold_validations(k=4)
    
    # Plot validation MAE
    boston_housing.plot_validation_mae()
    
    # Train the model with full training data
    print("Training the model on the entire dataset...")
    boston_housing.train(num_epochs=20)
    
    # Evaluate the model
    print("Evaluating the model on the test dataset...")
    boston_housing.evaluate()
