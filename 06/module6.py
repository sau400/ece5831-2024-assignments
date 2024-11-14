import numpy as np
import pickle
import sys
from PIL import Image
import matplotlib.pyplot as plt

# Define sigmoid and softmax functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # stability improvement
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Load the weights and biases from the pickle file
with open('raokhande_mnist_entire_model.pkl', 'rb') as f:
    params = pickle.load(f)

# Extract weights and biases from the loaded dictionary
params = params.params  # params is likely a dictionary
w1, b1, w2, b2 = params['w1'], params['b1'], params['w2'], params['b2']

# Predict function
def predict_image(image_path, w1, b1, w2, b2):
    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    image = image.resize((28, 28))  # Resize image to 28x28 pixels
    image_array = np.array(image)  # Convert to numpy array
    image_array = image_array.reshape(1, 28*28)  # Flatten the image
    image_array = image_array / 255.0  # Normalize the pixel values to [0, 1]

    # Forward pass (hidden layer)
    a1 = sigmoid(np.dot(image_array, w1) + b1)

    # Forward pass (output layer)
    a2 = softmax(np.dot(a1, w2) + b2)

    # Return the predicted class
    return np.argmax(a2), image_array

# Main function to run the prediction from the command line
if __name__ == "__main__":
    # Command line input: filename (e.g., '7.png')
    if len(sys.argv) != 3:
        print("Usage: python module5.py <path to sample> <digit>")
        sys.exit(1)

    image_path = sys.argv[1]  # Path to the image file
    true_digit = int(sys.argv[2])  # Expected digit (for comparison)
    
    # Predict the digit for the image
    predicted_digit, image_data = predict_image(image_path, w1, b1, w2, b2)
    
    # Display the image
    image = image_data.reshape(28, 28)  # Reshape back to 28x28 for visualization
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted Digit: {predicted_digit}")
    plt.show()

    # Print the predicted digit
    print(f"Predicted Digit: {predicted_digit}")
    
    # Check and print success/failure
    if predicted_digit == true_digit:
        print(f"Success: Image {image_path} is for digit {true_digit} and is recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image {image_path} is for digit {true_digit} but the inference result is {predicted_digit}.")
