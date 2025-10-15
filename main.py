from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
from tensorflow import keras
from PIL import Image
import os
import sys
import argparse


class DigitRecognizer:
    def __init__(self) -> None:
        self.model: Optional[keras.Model] = None
        self.load_or_train_model()

    def create_model(self) -> keras.Model:
        """
        Create a Convolutional Neural Network model for handwritten digit recognition.

        The model architecture consists of:
        - Reshape layer to convert 28x28 input to 28x28x1 format
        - Three convolutional layers with ReLU activation and max pooling
        - Flatten layer to convert to 1D
        - Dense layer with 64 neurons and ReLU activation
        - Dropout layer for regularization
        - Output layer with 10 neurons (for digits 0-9) and softmax activation

        Returns:
            keras.Model: Compiled CNN model ready for training
        """
        model = keras.Sequential([
            keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def load_or_train_model(self) -> None:
        """
        Load an existing trained model from disk or train a new one if none exists.

        Checks for a saved model file 'model.h5' in the current directory.
        If found, loads the model; otherwise, initiates training of a new model.
        """
        model_path = 'model.h5'

        if os.path.exists(model_path):
            print("Loading existing model...")
            self.model = keras.models.load_model(model_path)
        else:
            print("Training new model...")
            self.train_model()

    def train_model(self) -> None:
        """
        Train the CNN model on the MNIST handwritten digit dataset.

        Downloads and preprocesses the MNIST dataset, creates a new model,
        trains it for 5 epochs with validation split, evaluates performance,
        and saves the trained model to disk for future use.
        """
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        self.model = self.create_model()

        print("Training model...")
        history = self.model.fit(x_train, y_train,
                                 epochs=5,
                                 batch_size=128,
                                 validation_split=0.1,
                                 verbose=1)

        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")

        self.model.save('model.h5')
        print("Model saved!")

    def predict_digit(self, image_array: npt.NDArray[np.uint8]) -> Tuple[int, float]:
        """
        Predict the digit from a preprocessed image array.

        Takes a 28x28 grayscale image array, normalizes it, and uses the trained
        model to predict which digit (0-9) it represents along with confidence.

        Args:
            image_array: 28x28 numpy array representing a grayscale digit image

        Returns:
            Tuple containing:
                - predicted_digit (int): The predicted digit (0-9)
                - confidence (float): Prediction confidence (0.0-1.0)
        """
        image_array = image_array.reshape(1, 28, 28)
        image_array = image_array.astype('float32') / 255.0

        predictions = self.model.predict(image_array, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        return int(predicted_digit), float(confidence)


def center_image(img_array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Center the drawn digit within the 28x28 image boundaries.

    Finds the bounding box of the drawn content, extracts the digit,
    and repositions it in the center of a new 28x28 image to improve
    recognition accuracy by matching MNIST preprocessing.

    Args:
        img_array: Input 28x28 image array with the drawn digit

    Returns:
        Centered 28x28 image array with the digit positioned optimally
    """
    coords = np.column_stack(np.where(img_array > 30))
    if len(coords) == 0:
        return img_array

    top, left = coords.min(axis=0)
    bottom, right = coords.max(axis=0) + 1

    digit = img_array[top:bottom, left:right]

    centered = np.zeros((28, 28), dtype=np.uint8)

    h, w = digit.shape
    start_y = max(0, (28 - h) // 2)
    start_x = max(0, (28 - w) // 2)
    end_y = min(28, start_y + h)
    end_x = min(28, start_x + w)

    digit_h = end_y - start_y
    digit_w = end_x - start_x
    centered[start_y:end_y, start_x:end_x] = digit[:digit_h, :digit_w]

    return centered


def run_web_app(port: int) -> None:
    """Run the Flask web application."""
    try:
        from flask import Flask, render_template, request, jsonify
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
        import base64
        import io
    except ImportError:
        print("Flask and/or Flask-Limiter not installed. Install with: pip install flask flask-limiter")
        sys.exit(1)

    app = Flask(__name__)

    # Initialize rate limiter
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["1000 per hour"]
    )

    recognizer = DigitRecognizer()

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    @limiter.limit("30 per minute")
    def predict():
        try:
            data = request.get_json()
            image_data = data['image'].split(',')[1]  # Remove data:image/png;base64, prefix

            # Decode and process image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
            image_array = np.array(image)

            # Center image
            image_array = center_image(image_array)

            # Check if image has content
            if np.max(image_array) < 30:
                return jsonify({'error': 'Please draw a digit first!', 'digit': None, 'confidence': None})

            # Predict
            predicted_digit, confidence = recognizer.predict_digit(image_array)

            # Convert centered image back to base64 for frontend display
            from PIL import Image as PILImage
            centered_pil = PILImage.fromarray(image_array, mode='L')
            centered_buffer = io.BytesIO()
            centered_pil.save(centered_buffer, format='PNG')
            centered_base64 = base64.b64encode(centered_buffer.getvalue()).decode('utf-8')

            return jsonify({
                'digit': predicted_digit,
                'confidence': confidence,
                'error': None,
                'processed_image': f'data:image/png;base64,{centered_base64}'
            })

        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}', 'digit': None, 'confidence': None})

    print(f"Starting web server on http://localhost:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)


def main() -> None:
    """
    Initialize and run the handwritten digit recognition web application.

    Parses command line arguments for port configuration and starts the Flask server.
    """
    parser = argparse.ArgumentParser(description='Handwritten Digit Recognizer - Web Interface')
    parser.add_argument('--port', type=int, default=5000, help='Port for web interface (default: 5000)')
    args = parser.parse_args()

    print("Starting Handwritten Digit Recognition Web App...")
    run_web_app(args.port)


if __name__ == "__main__":
    main()
