import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# --- Badlav 1: Path Logic Fix ---
# Absolute path nikalne ke liye base_dir ka use
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "plant_model.tflite")
labels_path = os.path.join(base_dir, "labels.txt")

# Model Load Karein
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model Load Error: {e}")

# Labels Load Karein (Crash proof logic)
labels = []
if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    print("✅ Labels loaded successfully")
else:
    # Agar file nahi mili toh API crash nahi hogi, dummy labels use honge
    labels = [f"Disease_{i}" for i in range(15)]
    print(f"⚠️ Warning: labels.txt not found at {labels_path}. Using fallback labels.")

@app.route('/', methods=['GET'])
def home():
    return "API is Running! Use /predict endpoint for AI detection."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image uploaded"})
            
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
        
        confidence = float(output[0][idx])
        
        return jsonify({
            "status": "success",
            "prediction": labels[idx] if idx < len(labels) else f"Unknown_{idx}",
            "confidence": f"{confidence:.2%}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# --- Badlav 2: Render Port Binding (Safe Mode) ---
if __name__ == '__main__':
    # Render se PORT variable lega, agar nahi mila toh 10000 use karega
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)