import os
import json
import numpy as np

# Suppress noisy C++ tensorflow logs and oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models

def train_and_evaluate(epochs=10):
    print("=" * 60)
    print("Loading MNIST dataset...")
    print("=" * 60)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values from [0, 255] to [0.0, 1.0]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape images to (28, 28, 1) for Conv2D
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")
    print(f"Input shape: {x_train.shape[1:]}")

    # Build standard 2-convolution-layer CNN
    # Input -> Conv2D -> ReLU -> MaxPooling -> Conv2D -> ReLU -> MaxPooling -> Flatten -> Dense -> Dropout -> Dense(10) -> Softmax
    print("\nBuilding CNN model architecture...")
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(10, activation="softmax")
    ])

    model.summary()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )

    print("\nEvaluating model on the MNIST test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "digit_cnn.keras")
    model.save(model_path)
    print(f"\nModel saved successfully to {model_path}")

    # Store evaluated accuracy and metrics in metadata JSON for Flask to consume accurately
    # Display format: if >= 99% (0.99), formatted as "99%+", otherwise e.g. "98.8%"
    display_accuracy = "99%+" if test_accuracy >= 0.99 else f"{test_accuracy * 100:.1f}%"

    metadata = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "display_accuracy": display_accuracy,
        "epochs": epochs,
        "input_shape": [28, 28, 1],
        "num_classes": 10
    }

    metadata_path = os.path.join("models", "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata saved to {metadata_path} with display_accuracy: '{display_accuracy}'")
    print("=" * 60)

    return test_accuracy

if __name__ == "__main__":
    train_and_evaluate(epochs=10)

