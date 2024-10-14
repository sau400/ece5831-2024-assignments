import cv2
import numpy as np
import tensorflow as tf

# Function to load class names from labels.txt
def load_labels(label_file):
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

# Load the trained model from Teachable Machine
model = tf.keras.models.load_model('model/keras_model.h5')

# Load the class names from the labels.txt file
class_names = load_labels('model/labels.txt')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the image for model prediction (resize and normalize)
    img = cv2.resize(frame, (224, 224))  # Resize to model's input size
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    prediction_label = class_names[class_idx]
    confidence = predictions[0][class_idx] * 100  # Convert to percentage

    # Display the prediction on the frame
    text = f"{prediction_label} {confidence:.2f}%"  # Format the text
    cv2.putText(
        frame,  # The original frame (NumPy array)
        text,  # The text to display
        (50, 50),  # Position on the frame (x, y)
        cv2.FONT_HERSHEY_COMPLEX,  # Font type
        1,  # Font scale
        (0, 255, 0),  # Font color (BGR: Green)
        2  # Thickness of the text
    )

    # Display the frame
    cv2.imshow('Rock Paper Scissors', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


