import os
import numpy as np
import cv2

from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# ----------------- Flask Setup -----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ----------------- Load Model -----------------
MODEL_PATH = "mnist_model.h5"
model = load_model(MODEL_PATH)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "jfif"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ----------------- Preprocessing Function -----------------
def preprocess_image(image_path):
    """
    Reads an image from disk, converts to grayscale,
    resizes to 28x28, normalizes, and reshapes for the model.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))

    # Optional: uncomment this line if your digits are dark on light background
    # resized = 255 - resized

    norm = resized / 255.0
    # Your notebook used (1, 28, 28), so we keep same
    image_reshape = norm.reshape(1, 28, 28)

    return image_reshape, resized


# ----------------- Routes -----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template(
                "index.html",
                error="No file part in the request.",
                prediction=prediction,
                confidence=confidence,
                image_url=image_url,
            )

        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "index.html",
                error="No file selected. Please choose an image.",
                prediction=prediction,
                confidence=confidence,
                image_url=image_url,
            )

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            processed = preprocess_image(save_path)
            if processed is None:
                return render_template(
                    "index.html",
                    error="Could not read the uploaded image. Try another one.",
                    prediction=prediction,
                    confidence=confidence,
                    image_url=image_url,
                )

            image_reshape, resized = processed

            # Predict
            probs = model.predict(image_reshape)[0]
            pred_digit = int(np.argmax(probs))
            conf = float(np.max(probs) * 100)

            prediction = pred_digit
            confidence = round(conf, 2)
            image_url = url_for("static", filename=f"uploads/{filename}")

            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                image_url=image_url,
                error=None,
            )

        else:
            return render_template(
                "index.html",
                error="File type not allowed. Please upload an image file.",
                prediction=prediction,
                confidence=confidence,
                image_url=image_url,
            )

    # GET request
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        error=None,
    )


if __name__ == "__main__":
    # host='0.0.0.0' for deployment, debug=True for local dev
    app.run(debug=True)
