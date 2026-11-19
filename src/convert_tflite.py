import tensorflow as tf

MODEL_PATH = "models/waste_classifier.keras"
OUTPUT_PATH = "models/waste_classifier.tflite"

model = tf.keras.models.load_model(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"TensorFlow Lite model saved to {OUTPUT_PATH}")
