from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = "dataset/archive/Training"

train_data = ImageDataGenerator().flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

print(train_data.class_indices)