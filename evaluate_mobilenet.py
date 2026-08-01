import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ============================
# Load the Best MobileNet Model
# ============================
model = tf.keras.models.load_model("models/best_mobilenet_model.keras")

print("Model Loaded Successfully!")

# ============================
# Test Dataset
# ============================
test_path = "dataset/archive/Testing"

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# ============================
# Predictions
# ============================
predictions = model.predict(test_data)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_data.classes

class_labels = list(test_data.class_indices.keys())

# ============================
# Confusion Matrix
# ============================
cm = confusion_matrix(true_classes, predicted_classes)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(cm)

# ============================
# Classification Report
# ============================
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
))