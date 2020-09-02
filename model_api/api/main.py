from fastapi import FastAPI, Request, File, UploadFile
from cnn_model.predict import make_prediction
from PIL import Image
import io

app = FastAPI()


@app.get("/health")
async def health(request: Request):
    if request.method == 'GET':
        return "ok"
    
@app.post("/predict")
def predict(request: Request, file: UploadFile = File(...)):
    if request.method == 'POST':
        image = file.file
        print(type(image))
        contents = image.read()
        print(type(contents))
        image_jpeg = Image.open(io.BytesIO(contents))
        image_jpeg.save("input_image.jpg")
        prediction = float(make_prediction(input_image="input_image.jpg"))
        print(type(prediction))
        return {"prediction": prediction}
        