import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Dense, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ✅ Define dataset paths
DATASET_PATH = "D:/ayoko na/plant-identification-app/backend/plant_data/"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")

# ✅ Fixed Image Size (224x224) to match MobileNetV2 and ResNet50 requirements
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ✅ Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# ✅ Load train & validation data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

# ✅ Get class count
num_classes = len(train_generator.class_indices)
print(f"✅ Number of classes: {num_classes}")

# ✅ Corrected Input Layer Size (224x224)
input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))

# ✅ Load Pretrained Models (with correct input size)
base_model1 = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input_tensor)
base_model2 = ResNet50(weights="imagenet", include_top=False, input_tensor=input_tensor)

# ✅ Extract Features
feat1 = Flatten()(base_model1.output)
feat2 = Flatten()(base_model2.output)

# ✅ Merge Features
merged = Concatenate()([feat1, feat2])

# ✅ Final Classification Layer
output = Dense(num_classes, activation="softmax")(merged)

# ✅ Build Model
model = Model(inputs=input_tensor, outputs=output)

# ✅ Freeze Base Models Initially
for layer in base_model1.layers:
    layer.trainable = False
for layer in base_model2.layers:
    layer.trainable = False

# ✅ Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ Model Summary
model.summary()

# ✅ Train Model
EPOCHS = 10  # Adjust for speed vs accuracy
model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

# ✅ Unfreeze for Fine-Tuning
for layer in base_model1.layers:
    layer.trainable = True
for layer in base_model2.layers:
    layer.trainable = True

# ✅ Recompile with Lower Learning Rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# ✅ Continue Fine-Tuning
model.fit(train_generator, epochs=5, validation_data=val_generator)

# ✅ Save Fine-Tuned Model
model.save("D:/ayoko na/plant-identification-app/backend/plant_identification_finetuned.h5")
print("✅ Fine-tuned model saved!")
