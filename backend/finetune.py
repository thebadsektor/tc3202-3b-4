import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# ✅ Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ✅ Define paths
BASE_DIR = "D:/ayoko na/plant-identification-app/backend/plant_data/"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
MODEL_PATH = "D:/ayoko na/plant-identification-app/backend/model/plant_identification_model.h5"
FINE_TUNED_MODEL_PATH = "D:/ayoko na/plant-identification-app/backend/model/best_plant_identification_finetuned.h5"

# ✅ Check if dataset directories exist
if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    raise FileNotFoundError("❌ Training or validation directory not found!")

# ✅ Load pre-trained model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ✅ Unfreeze only the last few layers for fine-tuning
for layer in model.layers[:-10]:  # Freeze all but the last 10 layers
    layer.trainable = False

# ✅ Recompile model with lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Only rescale validation data

# ✅ Load dataset with corrected validation image size
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),  # Resize images to match model input
    batch_size=32,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),  # 🔥 FIXED: Ensure validation images match training size
    batch_size=32,
    class_mode="categorical"
)

# ✅ Callbacks for early stopping & best model saving
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(FINE_TUNED_MODEL_PATH, monitor="val_accuracy", save_best_only=True)

# ✅ Start fine-tuning
EPOCHS = 20  # Reduce epochs for faster training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, model_checkpoint]
)

print(f"✅ Fine-tuning complete! Model saved to {FINE_TUNED_MODEL_PATH}")
