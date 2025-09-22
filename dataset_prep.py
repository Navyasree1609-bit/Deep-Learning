import os
import cv2
import numpy as np

# Paths
DATA_DIR = "data"
IMG_SIZE = 64
OUTPUT_FILE = "dataset.npy"

# Classes
CATEGORIES = ["Parasitized", "Uninfected"]

def prepare_data():
    X = []  # features
    y = []  # labels
    
    # Check if data folder exists
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"❌ Data folder not found: {DATA_DIR}")
    
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        
        # Check if category folder exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Missing folder: {path}")
        
        print(f"📂 Loading {category} images...")
        
        for img_name in os.listdir(path):
            try:
                # Skip non-image files
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Could not read image: {img_name}")
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0  # normalize
                label = CATEGORIES.index(category)

                X.append(img)
                y.append(label)

            except Exception as e:
                print(f"⚠️ Error processing {img_name}: {e}")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"✅ Loaded {len(X)} images.")
    
    # Save tuple of (features, labels)
    np.savez(OUTPUT_FILE, X=X, y=y)
    print(f"💾 Dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_data()
