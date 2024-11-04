import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mnist import Mnist  # Ensure mnist.py (containing Mnist class) is in the same directory

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for prediction."""
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if img is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")

    # Convert to grayscale if the image is colored
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image to 28x28 pixels
    img = cv2.resize(img, (28, 28))

    # Apply binary thresholding to enhance features
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Normalize the image data to [0, 1] range
    img = img.astype(np.float32) / 255.0

    # Flatten the image to (784,) for the model input
    img = img.flatten()  # This creates a 1D array of 784 elements

    return img

if __name__ == '__main__':
    # Command-line arguments: <script> <image_path> <digit>
    if len(sys.argv) != 3:
        print("Usage: python module5.py <path to sample> <digit>")
        sys.exit(1)

    image_path = sys.argv[1]
    true_digit = int(sys.argv[2])

    # Load and display the image
    img = plt.imread(image_path)
    plt.imshow(img, cmap='gray')
    plt.title(f"True Digit: {true_digit}")
    plt.axis('off')  # Hide the axes
    plt.show()
    

    # Initialize and load the MNIST model
    mnist = Mnist()
    mnist.init_network()

    # Prepare the image for prediction
    img_input = load_and_preprocess_image(image_path) 
    print("Processed Image Shape:", img_input.shape)  # Should be (784,)

    # Make a prediction
    prediction = mnist.predict(img_input.reshape(1, -1))  # Reshape to (1, 784)
    predicted_digit = np.argmax(prediction)

    # Check and print the result
    if predicted_digit == true_digit:
        print(f"Success: Image {image_path} is for digit {true_digit} and is recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image {image_path} is for digit {true_digit} but the inference result is {predicted_digit}.")
