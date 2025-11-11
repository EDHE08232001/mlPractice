import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'knowledge', 'dataset'))

training_set_path = os.path.abspath(os.path.join(dataset_path, 'training_set'))
test_set_path = os.path.abspath(os.path.join(dataset_path, 'test_set'))
single_prediction_path = os.path.abspath(os.path.join(dataset_path, 'single_prediction'))
image_path = os.path.abspath(os.path.join(single_prediction_path, 'cat_or_dog_1.jpg'))

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

if not os.path.exists(training_set_path):
    raise FileNotFoundError(f"Training set path does not exist: {training_set_path}")

if not os.path.exists(test_set_path):
    raise FileNotFoundError(f"Test set path does not exist: {test_set_path}")

if not os.path.exists(single_prediction_path):
    raise FileNotFoundError(f"Single prediction path does not exist: {single_prediction_path}")

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image path does not exist: {image_path}")

print(f"[INFO] Dataset path verified: {dataset_path}")
print(f"[INFO] Training set path verified: {training_set_path}")
print(f"[INFO] Test set path verified: {test_set_path}")
print(f"[INFO] Single prediction path verified: {single_prediction_path}")
print(f"[INFO] Image path verified: {image_path}")