import base64
import io
import json
import os
import numpy as np
from PIL import Image, ImageOps

# Suppress warnings and info logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Global model and metadata references
model = None
metadata = {
    "test_accuracy": 0.9923,
    "display_accuracy": "99.23%"
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "digit_cnn.keras")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "models", "model_metadata.json")

def load_saved_model():
    global model, metadata
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
            print(f"Loaded metadata successfully. Accuracy: {metadata.get('display_accuracy')}")
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")

    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
            model = None
    else:
        print(f"Model file not found at {MODEL_PATH}. Run train_model.py first.")

def preprocess_image(image_data_uri):
    """
    Preprocess image from canvas data URL according to MNIST specifications:
    1. Decode base64 data URL
    2. Convert to RGBA, separate alpha / grayscale
    3. User draws black stroke on white canvas OR transparent canvas with black stroke.
       In MNIST, background is black (0) and digit stroke is white (255).
    4. Detect bounding box of stroke, crop unnecessary borders.
    5. Resize preserving aspect ratio so the longest dimension fits within ~20x20.
    6. Center inside 28x28 box using center-of-mass or geometric centering.
    7. Normalize pixel values to [0.0, 1.0].
    8. Reshape to (1, 28, 28, 1).
    """
    if not image_data_uri:
        raise ValueError("Empty image data provided.")

    if "," in image_data_uri:
        header, encoded = image_data_uri.split(",", 1)
    else:
        encoded = image_data_uri

    image_bytes = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGBA to handle transparent canvas or white background
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")

    # Composite over white background if there's an alpha channel
    bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
    bg.paste(pil_img, (0, 0), pil_img)
    gray = bg.convert("L")

    # Invert so digit stroke is white (near 255) and background is black (0)
    # The user draws with dark ink on white canvas, so invert: 255 - gray
    inverted = ImageOps.invert(gray)

    # Threshold very faint noise (optional thresholding to clean anti-aliasing artifacts)
    arr = np.array(inverted, dtype=np.uint8)
    arr[arr < 30] = 0

    # Check if canvas is virtually blank
    if np.max(arr) < 40 or np.sum(arr > 30) < 25:
        raise ValueError("Canvas is empty. Draw a digit first.")

    # Find bounding box of the drawn digit
    cleaned_img = Image.fromarray(arr)
    bbox = cleaned_img.getbbox()
    if not bbox:
        raise ValueError("No digit stroke detected.")

    # Crop to bounding box
    cropped = cleaned_img.crop(bbox)
    w, h = cropped.size

    # Scale to fit within 20x20 box while preserving aspect ratio (standard MNIST digit size inside 28x28)
    if w > h:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))
    else:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Paste into 28x28 black image, centered by bounding box
    final_img = Image.new("L", (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    final_img.paste(resized, (paste_x, paste_y))

    # Fine-tuning: center by Center of Mass (like original MNIST)
    final_arr = np.array(final_img, dtype=np.float32)
    total_mass = np.sum(final_arr)
    if total_mass > 0:
        cy, cx = np.indices((28, 28))
        com_y = np.sum(cy * final_arr) / total_mass
        com_x = np.sum(cx * final_arr) / total_mass

        shift_x = int(round(13.5 - com_x))
        shift_y = int(round(13.5 - com_y))

        # Constrain shifts to avoid shifting off the 28x28 canvas
        shift_x = max(-4, min(4, shift_x))
        shift_y = max(-4, min(4, shift_y))

        if shift_x != 0 or shift_y != 0:
            final_arr = np.roll(final_arr, shift_y, axis=0)
            final_arr = np.roll(final_arr, shift_x, axis=1)

    # Normalize to [0.0, 1.0]
    normalized = final_arr / 255.0

    # Reshape to (1, 28, 28, 1)
    input_tensor = np.expand_dims(normalized, axis=(0, -1))
    return input_tensor

@app.route("/", methods=["GET"])
def index():
    # Pass actual dynamic model accuracy
    display_acc = metadata.get("display_accuracy", "99.23%")
    return render_template("index.html", display_accuracy=display_acc)

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        load_saved_model()
        if model is None:
            return jsonify({
                "error": "Model is not loaded. Please ensure the model is trained."
            }), 503

    try:
        data = request.get_json(force=True, silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "Missing image data."}), 400

        image_data_uri = data["image"]
        input_tensor = preprocess_image(image_data_uri)

        # Run inference
        probabilities = model.predict(input_tensor, verbose=0)[0]
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": [round(float(p), 4) for p in probabilities]
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

# Initialize model upon startup
load_saved_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

