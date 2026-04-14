from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# 🔇 Optional: reduce TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI()

# ✅ Enable CORS (for React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model once
model = tf.keras.models.load_model("model.h5")

# ✅ Auto-detect input size from model
_, IMG_HEIGHT, IMG_WIDTH, _ = model.input_shape

# ✅ Your class names
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# ✅ Improved preprocessing (handles ANY image)
def preprocess_image(image: Image.Image):
    # Convert to RGB (handles grayscale, RGBA, etc.)
    image = image.convert("RGB")

    # Resize to model input size dynamically
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))

    # Convert to numpy + normalize
    image = np.array(image).astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

@app.get("/")
def home():
    return {"message": "Plant Disease Detection API is running 🌱"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # ✅ Open image safely
        image = Image.open(io.BytesIO(contents))

        # ✅ Preprocess
        processed_image = preprocess_image(image)

        # ✅ Predict
        predictions = model.predict(processed_image)
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        predicted_label = class_names[predicted_index]

        # ✅ Clean output
        crop, disease = predicted_label.split("___")

        return {
            "crop": crop.replace("_", " "),
            "disease": disease.replace("_", " "),
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}