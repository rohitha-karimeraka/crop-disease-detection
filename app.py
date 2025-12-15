from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

app = Flask(__name__)

# -----------------------------
# 1. Download model from Google Drive (only once)
# -----------------------------
MODEL_PATH = "crop_disease_cnn_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1350D2tIUDguykuVibtNpwlFuogbX5FvO"
    gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------
# 2. Load model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# 3. Class labels (folder order MUST match training)
# -----------------------------
class_names = [
    "Bacterial leaf blight",
    "Blight",
    "Brown spot",
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Leaf smut",
    "bacterial_blight",
    "curl_virus",
    "fussarium_wilt",
    "healthy"
]

# -----------------------------
# 4. Upload folder
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# 5. Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            img = Image.open(img_path).resize((224, 224))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            prediction = class_names[np.argmax(pred)]

    return render_template("index.html", prediction=prediction)

# -----------------------------
# 6. Run app
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
