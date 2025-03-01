from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
import gradio as gr
from pathlib import Path
import uvicorn
import io
from model_service import FashionClassifierService

app = FastAPI(title="Fashion Classifier API")

# Verificar si existe el directorio antes de montarlo
static_dir = Path("src/api/static")
if static_dir.exists() and static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Inicializar servicio del modelo
model_service = FashionClassifierService()

# Función de predicción para Gradio
def predict_image(image):
    """
    Función que procesa la imagen cargada por el usuario y devuelve las predicciones
    """
    if image is None:
        return {"Apparel": 0.0, "Accessories": 0.0, "Footwear": 0.0}
    
    # Convertir imagen de Gradio a bytes para el modelo existente
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Usar el servicio de modelo existente
    prediction = model_service.predict(img_byte_arr)
    
    # Devolver solo las probabilidades para el componente Label de Gradio
    return prediction['probabilities']

# Mantener el endpoint API original para compatibilidad con clientes existentes
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

# Crear la interfaz Gradio
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Cargar Imagen de Moda"),
    outputs=gr.Label(num_top_classes=3, label="Clasificación"),
    title="Fashion Classifier",
    description="Sube una imagen de un producto de moda para clasificarlo como Ropa (Apparel), Accesorios (Accessories), o Calzado (Footwear).",
    theme="huggingface",  # Tema moderno y atractivo
    allow_flagging="never"  # Desactivar el botón de flag
)

# Montar Gradio en FastAPI (reemplaza la página HTML anterior)
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)