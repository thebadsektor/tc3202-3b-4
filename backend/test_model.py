import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model_path = "D:/ayoko na/plant-identification-app/backend/model/best_plant_model.h5"
model = tf.keras.models.load_model(model_path)

# Define class labels (update based on your dataset)
class_labels = [
        'Cast Iron Plant (Aspidistra elatior)',
        'Jade plant (Crassula ovata)',
        'Chinese evergreen (Aglaonema)',
        'Schefflera',
        'Dumb Cane (Dieffenbachia spp.)',
        'Iron Cross begonia (Begonia masoniana)',
        'Anthurium (Anthurium andraeanum)',
        'Pothos (Ivy arum)',
        'Lilium (Hemerocallis)',
        'Tradescantia',
        'Dracaena',
        'Ctenanthe',
        'Asparagus Fern (Asparagus setaceus)',
        'Kalanchoe',
        'Poinsettia (Euphorbia pulcherrima)',
        'Bird of Paradise (Strelitzia reginae)',
        'English Ivy (Hedera helix)',
        'Orchid',
        'ZZ Plant (Zamioculcas zamiifolia)',
        'Areca Palm (Dypsis lutescens)',
        'Rubber Plant (Ficus elastica)',
        'Venus Flytrap',
        'African Violet (Saintpaulia ionantha)',
        'Birds Nest Fern (Asplenium nidus)',
        'Peace lily',
        'Rattlesnake Plant (Calathea lancifolia)',
        'Ponytail Palm (Beaucarnea recurvata)',
        'Calathea',
        'Parlor Palm (Chamaedorea elegans)',
        'Aloe Vera',
        'Elephant Ear (Alocasia spp.)',
        'Snake plant (Sanseviera)',
        'Lily of the valley (Convallaria majalis)',
        'Monstera Deliciosa (Monstera deliciosa)',
        'Tulip',
        'Prayer Plant (Maranta leuconeura)',
        'Hyacinth (Hyacinthus orientalis)',
        'Sago Palm (Cycas revoluta)',
        'Boston Fern (Nephrolepis exaltata)',
        'Yucca',
        'Christmas Cactus (Schlumbergera bridgesii)',
        'Chinese Money Plant (Pilea peperomioides)',
        'Begonia (Begonia spp.)',
        'Chrysanthemum',
        'Money Tree (Pachira aquatica)',
        'Daffodils (Narcissus spp.)',
        'Polka Dot Plant (Hypoestes phyllostachya)'
    ]

# Function to preprocess the image
def preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, target_size)  # Resize to match MobileNetV2 input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make a prediction
def predict(image_path):
    try:
        img = preprocess_image(image_path)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)  # Get class index
        confidence = np.max(predictions)  # Get confidence score

        print(f"Predicted: {class_labels[predicted_class]} ({confidence:.2f})")
    except Exception as e:
        print(f"Error: {e}")

# Test the model with a local image
if __name__ == "__main__":
    test_image = "D:/ayoko na/plant-identification-app/myplant.jpg"  # Change to your image path
    predict(test_image)
