from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import uvicorn
from model_service import FashionClassifierService

app = FastAPI(title="Fashion Classifier API")

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

# Inicializar servicio del modelo
model_service = FashionClassifierService()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return Path('src/api/static/index.html').read_text()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Verificar que el archivo es una imagen
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")
    
    # Leer contenido del archivo
    contents = await file.read()
    
    # Realizar predicción
    try:
        prediction = model_service.predict(contents)
        return prediction
    except Exception as e:
        raise HTTPException(500, f"Error al procesar la imagen: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)