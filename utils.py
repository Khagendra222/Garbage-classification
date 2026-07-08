import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ==========================
# Load Model
# ==========================
MODEL_PATH = "saved_model/best_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# Load Class Names
# ==========================
with open("saved_model/class_names.json", "r") as f:
    class_names = json.load(f)

# ==========================
# Waste Info
# ==========================
WASTE_INFO = {
    "cardboard": {
        "disposal": "Flatten and recycle in paper bin.",
        "tip": "Keep clean and dry."
    },
    "glass": {
        "disposal": "Rinse and recycle in glass bin.",
        "tip": "Handle carefully."
    },
    "metal": {
        "disposal": "Clean and recycle in metal bin.",
        "tip": "Saves energy."
    },
    "paper": {
        "disposal": "Place in paper recycling bin.",
        "tip": "Avoid wet paper."
    },
    "plastic": {
        "disposal": "Rinse and recycle plastic items.",
        "tip": "Reduce single-use plastic."
    },
    "trash": {
        "disposal": "Dispose in general waste.",
        "tip": "Separate recyclables."
    }
}

# ==========================
# Preprocessing (FIXED)
# ==========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image, dtype=np.float32)

    # 🔥 IMPORTANT FIX
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img

# ==========================
# Prediction
# ==========================
def predict_image(image):

    img = preprocess_image(image)
    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction) * 100)

    return {
        "category": predicted_class,
        "confidence": confidence,
        "disposal": WASTE_INFO[predicted_class]["disposal"],
        "tip": WASTE_INFO[predicted_class]["tip"]
    }