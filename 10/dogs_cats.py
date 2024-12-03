import pathlib
import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

CLASS_NAMES = ['dog', 'cat']
IMAGE_SHAPE = (180, 180, 3)
BATCH_SIZE = 32
BASE_DIR = pathlib.Path('dogs-vs-cats')
SRC_DIR = pathlib.Path('dogs-vs-cats-original/train')
EPOCHS = 20

class DogsCats:
    def __init__(self):
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.model = None

    def make_dataset_folders(self, subset_name, start_index, end_index):
        for category in CLASS_NAMES:
            dir = BASE_DIR / subset_name / category
            if not os.path.exists(dir):
                os.makedirs(dir)
            files = [f'{category}.{i}.jpg' for i in range(start_index, end_index)]
            for i, file in enumerate(files):
                shutil.copyfile(src=SRC_DIR / file, dst=dir / file)
                if i % 100 == 0:
                    print(f'src: {SRC_DIR / file} => dst: {dir / file}')

    def _make_dataset(self, subset_name):
        return image_dataset_from_directory(
            BASE_DIR / subset_name,
            image_size=IMAGE_SHAPE[:2],
            batch_size=BATCH_SIZE
        )

    def make_dataset(self):
        self.train_dataset = self._make_dataset('train')
        self.valid_dataset = self._make_dataset('validation')
        self.test_dataset = self._make_dataset('test')

    def build_network(self, augmentation=True):
        inputs = layers.Input(shape=IMAGE_SHAPE)
        x = inputs
        if augmentation:
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip('horizontal'),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2)
            ])
            x = data_augmentation(x)
        x = layers.Rescaling(1./255)(x)
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.AveragePooling2D(2)(x)
        x = layers.Conv2D(4, 3, activation='relu')(x)
        x = layers.AveragePooling2D(2)(x)
        x = layers.Conv2D(128, 3, activation='relu')(x)
        x = layers.AveragePooling2D(2)(x)
        x = layers.Conv2D(256, 3, activation='relu')(x)
        x = layers.AveragePooling2D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(200, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        self.model = models.Model(inputs, outputs)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, model_name):
        callbacks = [
            EarlyStopping(patience=5),
            ModelCheckpoint(filepath=f'{model_name}.keras'),
            TensorBoard(log_dir='./logs')
        ]
        history = self.model.fit(
            self.train_dataset,
            epochs=EPOCHS,
            validation_data=self.valid_dataset,
            callbacks=callbacks
        )
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(EPOCHS)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def load_model(self, model_name):
        self.model = tf.keras.models.load_model(f'{model_name}.keras')

    def predict(self, image_file):
        image = Image.open(image_file).convert('L')
        image = image.resize(IMAGE_SHAPE[:2])
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = image_np.reshape(1, *IMAGE_SHAPE)
        prediction = self.model.predict(image_np)
        predicted_class = CLASS_NAMES[int(prediction[0] > 0.5)]
        plt.imshow(image, cmap='gray')
        plt.title(f'Predicted: {predicted_class}')
        plt.axis('off')
        plt.show()
        print(f'Prediction: {predicted_class}')
