import os
from flask import Flask, request, render_template_string
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow import keras
import numpy as np

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load model
model = keras.models.load_model("models/malaria_cnn.keras")

# Prediction function
def predict_image(image_path):
    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return "Infected" if pred[0][0] > 0.5 else "Uninfected"

# Route allowing GET and POST
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        # Check if file is uploaded
        if "file" not in request.files:
            result = "No file selected"
        else:
            file = request.files["file"]
            if file.filename == "":
                result = "No file selected"
            else:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(filepath)
                result = predict_image(filepath)
    # Return clean HTML with only prediction
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Malaria Prediction</title>
        <style>
            body { font-family: Arial; text-align: center; padding: 50px; background: #f7f7f7; }
            .result { font-size: 36px; font-weight: bold; color: #333; margin-top: 50px; }
            input[type=file], input[type=submit] { font-size: 18px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required><br>
            <input type="submit" value="Predict">
        </form>
        {% if result %}
            <div class="result">{{ result }}</div>
        {% endif %}
    </body>
    </html>
    """, result=result)

if __name__ == "__main__":
    app.run(debug=False)
