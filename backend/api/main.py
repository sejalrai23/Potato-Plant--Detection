from fastapi  import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
# from PIL import Image
import tensorflow as tf
import requests

app= FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL =tf.keras.models.load_model('../models/1')
CLASS_NAMES=["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")

async def ping():
  return "Hello, Iam alive"


def read_file_as_numpy(data)->np.ndarray:
    image =np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")

async def predict(
  file: UploadFile = File(...)
):
    img=read_file_as_numpy(await file.read())
    img_batch=np.expand_dims(img,0)
    preds= MODEL.predict(img_batch)
    pred_class=CLASS_NAMES[np.argmax(preds[0])]
    conf=np.max(preds[0])
    return{
        'class': pred_class,
        'confidence': float(conf)
    }

if __name__ == "__main__":
  uvicorn.run(app, host='localhost', port=8000)