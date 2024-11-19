import cv2
import sys
import numpy as np
from le_net import LeNet

def process_single_image(image_path, expected_digit):
    """
    Load and process a single image, ensuring it is (28, 28) (grayscale),
    and compare the inference result with the expected digit.

    Args:
        image_path (str): Path to the image file.
        expected_digit (int): The digit expected to be recognized in the image.

    Returns:
        None: Outputs success or failure based on inference result.
    """
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Display the input image
    cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Input Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28
    resized_image = cv2.resize(grayscale_image, (28, 28))
    myimage = [resized_image]
    print(f"Processed image shape for inference: {resized_image.shape}")

    try:
        # Load the LeNet model
        LeNet_ = LeNet()
        LeNet_.load("D:\\pattern\\dataset\\dataset\\raokhande.keras")

        # Predict the digit
        inference_result = LeNet_.predict(myimage)

        # Check the inference result against the expected digit
        if int(inference_result[0]) == expected_digit:
            print(f"Success: Image {image_path} is for digit {expected_digit} recognized as {inference_result}.")
        else:
            print(f"Fail: Image {image_path} is for digit {expected_digit} but the inference result is {inference_result}.")
    except Exception as e:
        print(f"Error during model loading or inference: {e}")

if __name__ == "__main__":
    # Ensure correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python module5.py <image_filename> <expected_digit>")
        sys.exit(1)
    
    # Parse arguments
    image_filename = sys.argv[1]
    expected_digit = int(sys.argv[2])

    # Process the image
    process_single_image(image_filename, expected_digit)

