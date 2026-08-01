import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ===========================
# Load the trained model
# ===========================

model = tf.keras.models.load_model("models/brain_tumor_model.keras")

print("Model Loaded Successfully!\n")

# ===========================
# Testing Dataset Path
# ===========================

test_path = "dataset/archive/Testing"

# ===========================
# Load Testing Images
# ===========================

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# ===========================
# Predict All Images
# ===========================

predictions = model.predict(test_data)

# Convert probabilities to class index

predicted_classes = np.argmax(predictions, axis=1)

# True labels

true_classes = test_data.classes

# Class names

class_names = list(test_data.class_indices.keys())

# ===========================
# Confusion Matrix
# ===========================

cm = confusion_matrix(true_classes, predicted_classes)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================\n")

print(cm)

# ===========================
# Classification Report
# ===========================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================\n")

print(classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names
))
