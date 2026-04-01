
import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Model Load Karein
model_path = "plant_model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Labels Load Karein
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = Image.open(file).convert('RGB').resize((224, 224))
        
        # Preprocessing (0-1 normalize)
        input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        idx = np.argmax(output)
        
        return jsonify({
            "status": "success",
            "prediction": labels[idx],
            "confidence": float(output[0][idx])
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
