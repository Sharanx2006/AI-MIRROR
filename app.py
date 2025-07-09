from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
model = tf.keras.models.load_model("deepfake_model.h5")

def preprocess(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128)) / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/analyze', methods=['POST'])
def analyze():
    image = preprocess(request.files['file'].stream)
    prediction = model.predict(image)[0][0]
    label = "Real" if prediction < 0.5 else "Deepfake"
    confidence = round((1 - prediction) * 100, 2) if label == "Real" else round(prediction * 100, 2)
    return jsonify({'result': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
