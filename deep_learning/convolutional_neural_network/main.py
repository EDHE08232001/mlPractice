import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# ==========================================================
# 1. Resolve Dataset Path
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'knowledge', 'dataset'))

if not os.path.exists(dataset_path):
    print(f"[ERROR] Dataset path does not exist: {dataset_path}")
    sys.exit(1)
else:
    print(f"[INFO] Dataset path found: {dataset_path}")

# ==========================================================
# 2. Data Preparation
# ==========================================================
train_dir = os.path.join(dataset_path, 'training_set')
test_dir = os.path.join(dataset_path, 'test_set')

# Check directories
for dir_path in [train_dir, test_dir]:
    if not os.path.exists(dir_path):
        print(f"[ERROR] Directory missing: {dir_path}")
        sys.exit(1)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescale only for testing
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load datasets
training_set = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# ==========================================================
# 3. Build CNN Model
# ==========================================================
print(f"[INFO] TensorFlow version: {tf.__version__}")

cnn = tf.keras.models.Sequential(name="CatDogClassifier")

# Convolution + Pooling layer 1
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(64, 64, 3)
))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

# Convolution + Pooling layer 2
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu'
))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Fully connected layers
cnn.add(tf.keras.layers.Dense(128, activation='relu'))
cnn.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile model
cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("[INFO] CNN architecture summary:")
cnn.summary()

# ==========================================================
# 4. Training (Optional - uncomment if you want to train)
# ==========================================================
print("[INFO] Starting training...")
cnn.fit(
    training_set,
    validation_data=test_set,
    epochs=25
)
print("[INFO] Training complete.")

# ==========================================================
# 5. Single Image Prediction
# ==========================================================
image_path = os.path.join(dataset_path, 'single_prediction', 'cat_or_dog_1.jpg')
image_path = os.path.abspath(image_path)

if not os.path.exists(image_path):
    print(f"[ERROR] Test image does not exist: {image_path}")
    sys.exit(1)

test_image = load_img(image_path, target_size=(64, 64))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # normalize same as training

result = cnn.predict(test_image, verbose=0)

# Interpret prediction
prediction = 'dog' if result[0][0] > 0.5 else 'cat'
print(f"[INFO] The predicted class is: {prediction}")