import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMG_SIZE = 224  # Standard input size for many pre-trained models
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 47  # Based on the list you provided

# Define paths - update this to your specific path
BASE_DIR = "D:/ayoko na/plant-identification-app/backend/house_plant_species"
MODEL_PATH = "plant_identification_model.h5"

def prepare_data():
    """Prepare data generators for training, validation, and testing."""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # Use 20% of the data for validation
    )
    
    # Only rescaling for validation data
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = valid_datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Save class indices for later use
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())
    
    return train_generator, validation_generator, class_names

def build_model():
    """Build the model using transfer learning with MobileNetV2."""
    # Load the pre-trained model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_generator, validation_generator):
    """Train the model."""
    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    return history

def evaluate_model(model, validation_generator, class_names):
    """Evaluate the model and display results."""
    # Get predictions
    validation_generator.reset()
    y_pred = model.predict(validation_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true labels
    y_true = validation_generator.classes
    
    # Generate classification report
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print("Classification Report:\n", report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    return report

def fine_tune_model(model):
    """Fine-tune the model by unfreezing some layers of the base model."""
    # Unfreeze some layers of the base model
    for layer in model.layers[0].layers[-20:]:
        layer.trainable = True
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')

def create_sample_predictor(model, class_names):
    """Create a function to make predictions on a single image."""
    def predict_image(image_path):
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(IMG_SIZE, IMG_SIZE)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [(class_names[i], predictions[0][i] * 100) for i in top_indices]
        
        return {
            'prediction': class_names[predicted_class],
            'confidence': confidence,
            'top_predictions': top_predictions
        }
    
    return predict_image

def extract_scientific_name(plant_name):
    """Extract scientific name from folder name if available."""
    # Check if the plant name contains a scientific name in parentheses
    if '(' in plant_name and ')' in plant_name:
        return plant_name.split('(')[1].split(')')[0]
    return None

def main():
    """Main function to run the entire pipeline."""
    # Prepare data
    print("Preparing data...")
    train_generator, validation_generator, class_names = prepare_data()
    
    # Print class names for verification
    print(f"Found {len(class_names)} classes:")
    for i, class_name in enumerate(class_names):
        scientific_name = extract_scientific_name(class_name)
        print(f"{i+1}. {class_name} - Scientific name: {scientific_name}")
    
    # Build the model
    print("Building model...")
    model = build_model()
    
    # Train the model
    print("Training model...")
    history = train_model(model, train_generator, validation_generator)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    print("Evaluating model...")
    evaluate_model(model, validation_generator, class_names)
    
    # Fine-tune the model
    print("Fine-tuning model...")
    model = fine_tune_model(model)
    fine_tune_history = train_model(model, train_generator, validation_generator)
    
    # Plot fine-tuning history
    plot_training_history(fine_tune_history)
    
    # Evaluate the fine-tuned model
    print("Evaluating fine-tuned model...")
    evaluate_model(model, validation_generator, class_names)
    
    # Save the class names for later use
    with open('class_names.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    # Create a predictor function
    predictor = create_sample_predictor(model, class_names)
    
    # Example usage
    sample_image = "D:/ayoko na/plant-identification-app/backend/house_plant_species/African Violet (Saintpaulia ionantha)/1.jpg"
    if os.path.exists(sample_image):
        result = predictor(sample_image)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("Top 3 predictions:")
        for plant, confidence in result['top_predictions']:
            scientific_name = extract_scientific_name(plant)
            print(f"  {plant}: {confidence:.2f}% - Scientific name: {scientific_name}")

if __name__ == "__main__":
    main()