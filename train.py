import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
import os

# Load preprocessed dataset
data = np.load("dataset.npy.npz")  # Loading from .npz file
X = data["X"]
y = data["y"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build CNN Model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # fixed typo here
])

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Create models folder
os.makedirs("models", exist_ok=True)

# Save trained model
model.save("models/malaria_cnn.keras")
print("✅ Model saved to models/malaria_cnn.keras")
