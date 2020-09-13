from fastapi import FastAPI, Request, File, UploadFile
from cnn_model.predict import make_prediction
from cnn_model import __version__ as _version
from PIL import Image
import io

app = FastAPI()

@app.get("/")
async def root():
    return "home"

@app.get("/health")
async def health(request: Request):
    if request.method == 'GET':
        return "ok"
    
@app.get("/version")
async def version(request: Request):
    if request.method == 'GET':
        return {"model_version": _version,
                "api_version": "1.0.0"}
    
@app.post("/predict")
def predict(request: Request, file: UploadFile = File(...)):
    if request.method == 'POST':
        image = file.file
        contents = image.read()
        image_jpeg = Image.open(io.BytesIO(contents))
        image_jpeg.save("input_image.jpg")
        prediction = float(make_prediction(input_image="input_image.jpg"))
        return {"prediction": prediction}
        