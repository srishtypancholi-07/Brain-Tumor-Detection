import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("models/brain_tumor_model.keras")

# Class names (same order as training)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Path of image to predict
img_path = "test.jpg"   # Change this later

# Load image
img = image.load_img(img_path, target_size=(224, 224))

# Convert image to array
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

# Get predicted class
predicted_class = class_names[np.argmax(prediction)]

# Confidence
confidence = np.max(prediction) * 100

print("Prediction:", predicted_class)
print("Confidence: {:.2f}%".format(confidence))