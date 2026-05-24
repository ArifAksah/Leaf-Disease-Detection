import os
import numpy as np
import cv2
from tensorflow import keras
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from supabase import create_client, Client
import datetime
import uvicorn
import uuid

load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# [CORE JEMBATAN API: Inisialisasi FastAPI]
app = FastAPI(title="Leaf Disease Detection API")

# [CORE JEMBATAN API: CORS] Membuka akses agar React (Frontend) diizinkan mengambil data dari server ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Model
model = keras.models.load_model('CNNModel.keras')

@app.get('/')
def health_check():
    return {
        'status': 'ok',
        'message': 'Leaf Disease Detection API is running',
        'endpoint': '/predict',
        'method': 'POST'
    }

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

# [CORE JEMBATAN API: Endpoint] URL /predict yang akan dipanggil oleh React
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    try:
        # Read file asynchronously
        contents = await file.read()
        file_bytes = np.asarray(bytearray(contents), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if opencv_image is None:
            raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid JPG/PNG image.")

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
        
        # Save to Supabase
        if supabase:
            image_url = None
            unique_filename = None
            
            try:
                # 1. Generate unique filename
                file_extension = file.filename.split(".")[-1]
                unique_filename = f"{uuid.uuid4()}.{file_extension}"
                
                # 2. Upload to Storage (menggunakan bucket leaf_images)
                supabase.storage.from_("leaf_images").upload(
                    file=contents,
                    path=unique_filename,
                    file_options={"content-type": file.content_type}
                )
                
                # 3. Get Public URL
                image_url = supabase.storage.from_("leaf_images").get_public_url(unique_filename)
                
            except Exception as storage_error:
                print(f"Failed to upload image to Storage: {storage_error}")

            try:
                supabase.table("predictions").insert({
                    "disease_id": predicted_class,
                    "confidence": confidence,
                    "image_url": image_url,
                    "created_at": datetime.datetime.utcnow().isoformat()
                }).execute()
            except Exception as db_error:
                print(f"Failed to insert into Database: {db_error}")
                # Rollback: Hapus file dari storage jika gagal insert ke DB
                if unique_filename:
                    supabase.storage.from_("leaf_images").remove([unique_filename])
        
        # [CORE JEMBATAN API: Pengiriman Kembali] Membungkus hasil akhir Keras/AI ke JSON
        return {
            'success': True,
            'disease_id': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    uvicorn.run("api:app", host='0.0.0.0', port=port)
