from fastapi import FastAPI, UploadFile, File
from predict_with_prevention import predict_disease

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.jpg", "wb") as f:
        f.write(contents)

    predicted_class, preventions = predict_disease("temp.jpg")
    return {
        "disease": predicted_class,
        "prevention": preventions
    }
