from typing import Tuple, Optional
import tkinter as tk
from tkinter import messagebox
import numpy as np
import numpy.typing as npt
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageDraw
import os
import time


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
            layers.Reshape((28, 28, 1), input_shape=(28, 28)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
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

        self.model.save('digit_model.h5')
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


class DrawingApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("500x600")

        self.recognizer = DigitRecognizer()

        self.canvas_size: int = 280
        self.brush_size: int = 15
        self.drawing: bool = False
        self.last_prediction_time: float = 0.0
        self.prediction_delay: float = 0.2  # 200ms delay between predictions
        self.last_x: Optional[int] = None
        self.last_y: Optional[int] = None

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas: tk.Canvas
        self.result_label: tk.Label
        self.confidence_label: tk.Label

        self.setup_ui()

    def setup_ui(self) -> None:
        """
        Set up the complete user interface for the digit recognition application.

        Creates and configures all UI elements including:
        - Title label
        - Drawing canvas with mouse event bindings
        - Clear button
        - Result display labels
        """
        title_label = tk.Label(self.root, text="Draw a digit (0-9)",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        self.canvas = tk.Canvas(self.root, width=self.canvas_size,
                                height=self.canvas_size, bg='black',
                                cursor='pencil')
        self.canvas.pack(pady=10)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_digit)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        clear_btn = tk.Button(button_frame, text="Clear",
                              command=self.clear_canvas,
                              font=("Arial", 12), width=10)
        clear_btn.pack(side=tk.LEFT, padx=5)

        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=20)

        self.result_label = tk.Label(result_frame, text="Draw a digit to get prediction",
                                     font=("Arial", 14))
        self.result_label.pack()

        self.confidence_label = tk.Label(result_frame, text="",
                                         font=("Arial", 12), fg='gray')
        self.confidence_label.pack()

    def start_draw(self, event: tk.Event) -> None:
        """
        Initialize drawing mode when mouse button is pressed.

        Sets the drawing flag to True and records the initial position,
        enabling smooth line drawing with interpolation.

        Args:
            event: Tkinter mouse event object containing coordinates
        """
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw_digit(self, event: tk.Event) -> None:
        """
        Draw on both the display canvas and internal PIL image during mouse motion.

        Creates smooth continuous lines by interpolating between the previous
        and current mouse positions to eliminate gaps. Also triggers real-time
        prediction with throttling to avoid performance issues.

        Args:
            event: Tkinter mouse event object containing current cursor coordinates
        """
        if self.drawing and self.last_x is not None and self.last_y is not None:
            x, y = event.x, event.y

            # Draw line on tkinter canvas for smooth appearance
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=self.brush_size, fill='white',
                                    capstyle=tk.ROUND, smooth=tk.TRUE)

            # Draw line on PIL image for recognition
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill=255, width=self.brush_size)

            # Update last position
            self.last_x, self.last_y = x, y

            # Real-time prediction with throttling
            current_time = time.time()
            if current_time - self.last_prediction_time > self.prediction_delay:
                self.predict_digit()
                self.last_prediction_time = current_time

    def end_draw(self, event: tk.Event) -> None:
        """
        Finalize drawing when mouse button is released.

        Sets drawing flag to False, resets position tracking, and automatically 
        triggers prediction after each drawing stroke is completed.

        Args:
            event: Tkinter mouse event object
        """
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.predict_digit()

    def clear_canvas(self) -> None:
        """
        Reset the drawing canvas and internal image to blank state.

        Removes all drawings from the display canvas, creates a new blank
        PIL image, resets position tracking, and resets the result labels.
        """
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.result_label.config(text="Draw a digit to get prediction")
        self.confidence_label.config(text="")

    def preprocess_image(self) -> npt.NDArray[np.uint8]:
        """
        Convert the drawn image to the 28x28 format required by the model.

        Resizes the canvas image to 28x28 pixels using high-quality resampling,
        converts to numpy array, and applies centering to better match the
        MNIST dataset format.

        Returns:
            28x28 numpy array representing the preprocessed digit image
        """
        resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(resized)
        img_array = self.center_image(img_array)

        return img_array

    def center_image(self, img_array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
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

    def predict_digit(self) -> None:
        """
        Process the drawn image and display the predicted digit with confidence.

        Preprocesses the current drawing, validates that something was drawn,
        calls the recognition model for prediction, and updates the UI with
        the results. Handles errors gracefully with user-friendly messages.
        """
        try:
            processed_image = self.preprocess_image()

            if np.max(processed_image) < 30:
                self.result_label.config(text="Please draw a digit first!")
                self.confidence_label.config(text="")
                return

            predicted_digit, confidence = self.recognizer.predict_digit(
                processed_image)

            self.result_label.config(text=f"Predicted Digit: {predicted_digit}",
                                     font=("Arial", 18, "bold"))
            self.confidence_label.config(text=f"Confidence: {confidence:.2%}")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")


def main() -> None:
    """
    Initialize and run the handwritten digit recognition application.

    Creates the main Tkinter window, initializes the DrawingApp,
    and starts the GUI event loop. Provides console feedback about
    the application startup process.
    """
    print("Starting Handwritten Digit Recognizer...")

    root = tk.Tk()

    app = DrawingApp(root)

    print("Application ready! Draw digits in the canvas.")
    root.mainloop()


if __name__ == "__main__":
    main()
