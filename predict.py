import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ['KMP_WARNINGS'] = '0'         # Suppress oneDNN/MKL warnings

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load trained model
model = load_model("models/malaria_cnn.keras")

def preprocess_image(image_path, target_size=(64, 64)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_cell(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array, verbose=0)[0][0]
    return "INFECTED" if prediction > 0.5 else "UNINFECTED"

# Example usage
image_path = r"C:\Users\pocha\OneDrive\Documents\malaria\data\Uninfected\C1_thinF_IMG_20150604_104722_cell_9.png"
print(predict_cell(image_path))
