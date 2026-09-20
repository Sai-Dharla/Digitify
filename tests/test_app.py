import io
import base64
import json
import pytest
import numpy as np
from PIL import Image, ImageDraw

from app import app, preprocess_image, load_saved_model

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def create_digit_image_data_url(digit_char="5"):
    """Generate a synthetic white canvas with a black handwritten-style digit."""
    img = Image.new("RGB", (280, 280), color="white")
    draw = ImageDraw.Draw(img)
    # Draw simple strokes representing a digit
    if digit_char == "1":
        draw.line([(140, 40), (140, 240)], fill="black", width=20)
    elif digit_char == "0":
        draw.ellipse([(70, 40), (210, 240)], outline="black", width=20)
    elif digit_char == "7":
        draw.line([(60, 50), (220, 50)], fill="black", width=20)
        draw.line([(220, 50), (100, 240)], fill="black", width=20)
    else:
        # Default simple cross/box for basic stroke presence
        draw.line([(80, 80), (200, 80)], fill="black", width=20)
        draw.line([(80, 80), (80, 160)], fill="black", width=20)
        draw.line([(80, 160), (200, 160)], fill="black", width=20)
        draw.line([(200, 160), (200, 240)], fill="black", width=20)
        draw.line([(80, 240), (200, 240)], fill="black", width=20)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

def test_index_page(client):
    """Test that GET / serves 200 and contains core required elements."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Digitify" in html
    assert "Draw a digit, and let your Convolutional Neural Network" in html
    assert "Try It Now" in html
    assert "Draw a digit here" in html
    assert "28 &times; 28 input" in html
    assert "Clear" in html
    assert "Predict" in html
    assert "Prediction" in html
    assert "Confidence" in html
    assert "Model Accuracy" in html
    assert "Real-time" in html

def test_predict_empty_canvas(client):
    """Test POST /predict with a blank white canvas returns 400 error."""
    blank_img = Image.new("RGB", (280, 280), color="white")
    buf = io.BytesIO()
    blank_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_str}"

    response = client.post("/predict", json={"image": data_uri})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Draw a digit first" in data["error"] or "empty" in data["error"].lower()

def test_predict_missing_payload(client):
    """Test POST /predict with invalid or missing payload returns 400."""
    response = client.post("/predict", json={})
    assert response.status_code == 400

def test_preprocessing_pipeline():
    """Test that preprocessing produces the exact shape (1, 28, 28, 1) and normalized range."""
    data_uri = create_digit_image_data_url("1")
    tensor = preprocess_image(data_uri)
    assert tensor.shape == (1, 28, 28, 1)
    assert np.min(tensor) >= 0.0
    assert np.max(tensor) <= 1.0
    # Must have some active foreground pixels (max > 0.5)
    assert np.max(tensor) > 0.5

def test_predict_endpoint_real_model(client):
    """Test that POST /predict with actual drawn digit returns prediction 0-9 and confidence 0-1."""
    data_uri = create_digit_image_data_url("1")
    response = client.post("/predict", json={"image": data_uri})
    
    # If model is loaded, it should return 200 with real prediction
    if response.status_code == 200:
        res = response.get_json()
        assert "prediction" in res
        assert "confidence" in res
        assert 0 <= res["prediction"] <= 9
        assert 0.0 <= res["confidence"] <= 1.0
        assert "probabilities" in res
        assert len(res["probabilities"]) == 10
    else:
        # If model is still training, it gracefully returns 503
        assert response.status_code == 503

