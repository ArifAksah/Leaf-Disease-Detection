import numpy as np
import cv2
from keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS

# [CORE JEMBATAN API: Inisialisasi]
app = Flask(__name__)
# [CORE JEMBATAN API: CORS] Membuka akses agar React (Frontend) diizinkan mengambil data dari server ini
CORS(app)

# Load the Model
# It's better to load the model once when the server starts
model = load_model('CNNModel.h5', compile=False)

# Name of Classes
CLASS_NAMES = [
    'Tomato-Early_Bright',
    'Tomato-Healthy',
    'Tomato-Late_bright',
    'Tomato-Leaf_Mold',
    'Tomato-Septoria_LeafSpot',
    'Tomato-Spider_Mites',
    'Tomato-Target_Spot',
    'Tomato-YellowLeaf-CurlVirus',
    'Tomato-Bacterial_spot',
    'Tomato-mosaic_virus'
]

# [CORE JEMBATAN API: Endpoint] Menentukan URL /predict yang akan dipanggil oleh React (menggunakan metode POST)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    # [CORE JEMBATAN API: Penerimaan Menerima Teks] Menangkap gambar fisik yang baru saja di-*upload* pengguna lewat Frontend React
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if opencv_image is None:
            return jsonify({'error': 'Invalid image file. Please upload a valid JPG/PNG image.'}), 400

        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (224, 224))

        # Ensure image has 3 channels
        if len(opencv_image.shape) == 2:
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_GRAY2BGR)
        elif opencv_image.shape[2] == 4:
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGRA2BGR)

        # Convert image to 4 Dimension
        opencv_image = np.expand_dims(opencv_image, axis=0)

        # Make Prediction
        Y_pred = model.predict(opencv_image)[0] # Get the first (and only) result array
        
        # Find the class with highest probability
        max_idx = np.argmax(Y_pred)
        predicted_class = CLASS_NAMES[max_idx]
        confidence = float(Y_pred[max_idx]) * 100
        
        # Create a list of all predictions
        all_predictions = []
        for i, class_name in enumerate(CLASS_NAMES):
            all_predictions.append({
                'disease_id': class_name,
                'confidence': float(Y_pred[i]) * 100
            })
            
        # Sort predictions by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # [CORE JEMBATAN API: Pengiriman Kembali] Membungkus hasil akhir Keras/AI ke bahasa JSON agar bisa dibaca dan ditampilkan secara visual oleh React
        return jsonify({
            'success': True,
            'disease_id': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
