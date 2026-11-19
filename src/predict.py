import tensorflow as tf
import numpy as np
import sys

MODEL_PATH = "models/waste_classifier.keras"
CLASS_NAMES_PATH = "models/class_names.txt"

IMG_SIZE = (224, 224)

if len(sys.argv) < 2:
    print("Usage:")
    print("python src/predict.py path/to/image.jpg")
    sys.exit()

image_path = sys.argv[1]

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

image = tf.keras.utils.load_img(
    image_path,
    target_size=IMG_SIZE
)

image_array = tf.keras.utils.img_to_array(image)

image_array = tf.expand_dims(
    image_array,
    axis=0
)

prediction = model.predict(
    image_array,
    verbose=0
)[0]

index = np.argmax(prediction)

label = class_names[index]
confidence = prediction[index] * 100

print("\nPrediction")
print("----------------")
print(f"Class: {label}")
print(f"Confidence: {confidence:.2f}%")
