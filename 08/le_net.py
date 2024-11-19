from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os

class LeNet:
    def __init__(self, batch_size=32, epochs=20):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self._create_lenet()
        self._compile()

    def _create_lenet(self):
    
        self.model = Sequential([
            Conv2D(filters=6, kernel_size=(5, 5), activation='sigmoid',
                   input_shape=(28, 28, 1), padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),
            
            Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid', padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),
            
            Flatten(),
            Dense(120, activation='sigmoid'),
            Dense(84, activation='sigmoid'),
            Dense(10, activation='softmax')
        ])

    def _compile(self):
        """
        Compile the LeNet model with an Adam optimizer and categorical crossentropy loss.
        """
        if self.model is None:
            raise ValueError('Error: Model is not created yet.')
        
        self.model.compile(optimizer='Adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def _preprocess(self):
        """
        Preprocess the MNIST dataset: normalize, reshape, and one-hot encode the labels.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Normalize the pixel values to the range [0, 1]
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        
        # Add channel dimension
        self.x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        self.x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        
        # One-hot encode the labels
        self.y_train = to_categorical(y_train, 10)
        self.y_test = to_categorical(y_test, 10)

    def train(self):
        """
        Train the model using the preprocessed MNIST dataset.
        """
        self._preprocess()
        self.model.fit(self.x_train, self.y_train, 
                       batch_size=self.batch_size, 
                       epochs=self.epochs,
                       validation_data=(self.x_test, self.y_test))

    def save(self, model_path_name):
        """
        Save the trained model to the specified file path.
        """
        if not model_path_name.endswith(".keras"):
            model_path_name += ".keras"
        self.model.save(model_path_name)
        print(f"Model saved as {model_path_name}")

    def load(self, model_path_name):
        """
        Load a saved model from the specified file path.
        """
        if not os.path.exists(model_path_name):
            raise FileNotFoundError(f"No such file: {model_path_name}")
        self.model = load_model(model_path_name)
        print(f"Model loaded from {model_path_name}")

    def predict(self, images):
        """
        Predict the class probabilities for a list of images.
        """
        if self.model is None:
            raise ValueError("Model is not loaded or created yet.")
        
        # Preprocess images: normalize and add channel dimension
        images = [img / 255.0 for img in images]
        images = [img.reshape(1, 28, 28, 1) for img in images]
        
        # Run predictions and return results
        predictions = [self.model.predict(img, verbose=0) for img in images]
        return [pred.argmax() for pred in predictions]


if __name__ == "__main__":
    # Example usage
    lenet = LeNet(batch_size=64, epochs=10)
    lenet.train()
    lenet.save("raokhande")
    lenet.load("raokhande.keras")
 
