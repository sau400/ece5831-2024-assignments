from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import sys
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')  # Set the backend to TkAgg
# import matplotlib.pyplot as plt


def load_image(file_path):
    # Replace this with the path to your image
    image = Image.open(file_path).convert("RGB")
    return image


def init():
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)


def load_my_model():
    # Load the model
    model = load_model("model/keras_model.h5")

    # Load the labels
    class_names = open("model/labels.txt", "r").readlines()

    return model, class_names

def prep_input(image):

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    return data


def predict(model, class_names, data):
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    print(prediction)


if __name__ == "__main__":

    file_path = sys.argv[1]

    init()
    image = load_image(file_path)
    model, class_names = load_my_model()
    data = prep_input(image)
    predict(model, class_names, data)
    plt.imshow(image)
    plt.show()


