from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained CNN model
model = tf.keras.models.load_model("crop_disease_cnn_model.h5")

# Class labels (mee dataset folders order lo)
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

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            # Image preprocessing
            img = Image.open(img_path).resize((224, 224))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            # Prediction
            pred = model.predict(img)
            prediction = class_names[np.argmax(pred)]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    aapp.run(host="0.0.0.0", port=5000)

