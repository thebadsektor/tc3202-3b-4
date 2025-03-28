from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import traceback
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Log TensorFlow version to check compatibility
print(f"✅ TensorFlow Version: {tf.__version__}")

# ✅ Define class names (Ensure order matches model training)
CLASS_NAMES = [
    "African Violet (Saintpaulia ionantha)", "Aloe Vera", "Anthurium (Anthurium andraeanum)", 
    "Areca Palm (Dypsis lutescens)", "Asparagus Fern (Asparagus setaceus)", "Begonia (Begonia spp.)", 
    "Bird of Paradise (Strelitzia reginae)", "Birds Nest Fern (Asplenium nidus)", 
    "Boston Fern (Nephrolepis exaltata)", "Calathea", "Cast Iron Plant (Aspidistra elatior)", 
    "Chinese Money Plant (Pilea peperomioides)", "Chinese evergreen (Aglaonema)", 
    "Christmas Cactus (Schlumbergera bridgesii)", "Chrysanthemum", "Ctenanthe", 
    "Daffodils (Narcissus spp.)", "Dracaena", "Dumb Cane (Dieffenbachia spp.)", 
    "Elephant Ear (Alocasia spp.)", "English Ivy (Hedera helix)", "Hyacinth (Hyacinthus orientalis)", 
    "Iron Cross begonia (Begonia masoniana)", "Jade plant (Crassula ovata)", "Kalanchoe", 
    "Lilium (Hemerocallis)", "Lily of the valley (Convallaria majalis)", "Money Tree (Pachira aquatica)", 
    "Monstera Deliciosa (Monstera deliciosa)", "Orchid", "Parlor Palm (Chamaedorea elegans)", 
    "Peace lily", "Poinsettia (Euphorbia pulcherrima)", "Polka Dot Plant (Hypoestes phyllostachya)", 
    "Ponytail Palm (Beaucarnea recurvata)", "Pothos (Ivy arum)", "Prayer Plant (Maranta leuconeura)", 
    "Rattlesnake Plant (Calathea lancifolia)", "Rubber Plant (Ficus elastica)", 
    "Sago Palm (Cycas revoluta)", "Schefflera", "Snake plant (Sanseviera)", "Tradescantia", 
    "Tulip", "Venus Flytrap", "Yucca", "ZZ Plant (Zamioculcas)"
]

# ✅ Load model
MODEL_PATH = "D:/ayoko na/plant-identification-app/tc3202-3b-4/backend/model/plant_identification_model.h5"

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print(f"✅ Model expects input shape: {model.input_shape}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(traceback.format_exc())
    exit(1)  # Stop execution if model fails to load

# ✅ Image preprocessing function
def preprocess_image(image):
    try:
        input_shape = model.input_shape
        expected_height = input_shape[1] if input_shape[1] is not None else 150
        expected_width = input_shape[2] if input_shape[2] is not None else 150
        print(f"✅ Resizing image to {expected_width}x{expected_height}")

        image = image.resize((expected_width, expected_height))
        img_array = np.array(image) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

# ✅ Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print(f"✅ CLASS_NAMES: {CLASS_NAMES}")  # Debugging

        if not CLASS_NAMES:
            return jsonify({"error": "Class names not loaded properly"}), 500

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        print(f"✅ Received file: {file.filename}")

        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        print(f"✅ Original image size: {image.size}")

        processed_image = preprocess_image(image)
        if processed_image is None:
            return jsonify({"error": "Image processing failed"}), 500

        print("✅ Making prediction...")
        prediction = model.predict(processed_image)

        predicted_class_index = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class_index]) * 100

        if predicted_class_index >= len(CLASS_NAMES):
            return jsonify({"error": f"Predicted index {predicted_class_index} is out of range"}), 500

        top_indices = np.argsort(prediction[0])[-3:][::-1]
        top_predictions = [{"plant_name": CLASS_NAMES[idx], "accuracy": round(float(prediction[0][idx]) * 100, 2)} for idx in top_indices]

        return jsonify({
            "predicted_plant": CLASS_NAMES[predicted_class_index],
            "accuracy": round(confidence, 2),
            "top_predictions": top_predictions,
            "message": f"Identified plant as {CLASS_NAMES[predicted_class_index]} with {round(confidence, 2)}% accuracy"
        })

    except Exception as e:
        print(f"❌ Error in /predict route: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ✅ Login route
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    
    # Simple hardcoded authentication (replace with proper authentication in production)
    if username == "admin" and password == "password":
        return jsonify({"status": "success", "message": "Login successful"}), 200
    else:
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401

# ✅ Health check route
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Server is running"})

# ✅ Run Flask server
if __name__ == "__main__":
    print("✅ Starting plant identification server...")
    app.run(debug=True, host="0.0.0.0", port=5000)