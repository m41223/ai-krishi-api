import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# --- Badlav 1: Model aur Labels ke liye Dynamic Path ---
# Ye ensures karta hai ki file hamesha mil jaye
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "plant_model.tflite")
labels_path = os.path.join(base_dir, "labels.txt")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image uploaded"})
            
        file = request.files['image']
        img = Image.open(file).convert('RGB').resize((224, 224))
        
        input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        idx = np.argmax(output)
        
        # Confidence score ko percentage mein dikhane ke liye
        confidence = float(output[0][idx])
        
        return jsonify({
            "status": "success",
            "prediction": labels[idx],
            "confidence": f"{confidence:.2%}" # Example: 98.50%
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# --- Badlav 2: Render Port Binding ---
if __name__ == '__main__':
    # Render environmental variable 'PORT' use karta hai
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)