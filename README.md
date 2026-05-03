---
title: Leaf Disease Detection API
emoji: 🍃
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
---

# 🍃 Leaf Disease Detection API

API Flask untuk deteksi penyakit tanaman tomat menggunakan CNN (Convolutional Neural Network).

## Endpoint

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Upload image untuk prediksi |

## Contoh Request

```bash
curl -X POST -F "file=@gambar.jpg" https://your-space.hf.space/predict
```

## Response Format

```json
{
  "success": true,
  "disease_id": "Tomato-Early_Bright",
  "confidence": 95.5,
  "all_predictions": [...]
}
```

## Model Classes

- Tomato-Early_Bright
- Tomato-Healthy
- Tomato-Late_bright
- Tomato-Leaf_Mold
- Tomato-Septoria_LeafSpot
- Tomato-Spider_Mites
- Tomato-Target_Spot
- Tomato-YellowLeaf-CurlVirus
- Tomato-Bacterial_spot
- Tomato-mosaic_virus

## Catatan

API ini digunakan oleh frontend React yang di-deploy di Vercel.