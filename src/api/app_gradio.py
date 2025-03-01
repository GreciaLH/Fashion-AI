import gradio as gr
import torch
import sys
import os
from pathlib import Path
from PIL import Image
import io

# Agregar rutas necesarias
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
from model_service import FashionClassifierService

# Inicializar servicio del modelo
model_service = FashionClassifierService()

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
    
    # Formatear los resultados para Gradio
    probabilities = prediction['probabilities']
    
    return probabilities

# Configurar la interfaz de Gradio
# Configuración de la interfaz
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Sube tu imagen de moda 👗"),
    outputs=gr.Label(num_top_classes=3, label="Resultados"),
    theme=gr.themes.Soft(
        primary_hue="purple",
        bg_color="#0d0d0d",  # Fondo oscuro para resaltar efectos de vidrio
        font=[gr.themes.GoogleFont("Montserrat"), "sans-serif"]  # Corrige "Montserrat" (falta una 't')
    ),
    examples=[["ejemplo_camiseta.jpg"], ["ejemplo_zapato.jpg"]]
)

# Iniciar la aplicación Gradio
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)