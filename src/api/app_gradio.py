import gradio as gr
import torch
import sys
import os
from pathlib import Path
from PIL import Image
import io
import shutil
import glob

# Agregar rutas necesarias
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
from model_service import FashionClassifierService

# Configurar directorio de ejemplos
EXAMPLES_DIR = Path(__file__).parent / "examples"
EXAMPLES_DIR.mkdir(exist_ok=True)

# Rutas a las imágenes de ejemplo
EXAMPLE_ROPA = EXAMPLES_DIR / "ejemplo_ropa.jpg"
EXAMPLE_ACCESORIOS = EXAMPLES_DIR / "ejemplo_accesorios.jpg"
EXAMPLE_CALZADO = EXAMPLES_DIR / "ejemplo_calzado.jpg"

def get_model_service():
    """Función para obtener el servicio del modelo (facilita el mockeo en pruebas)"""
    return FashionClassifierService()

# Inicializar servicio del modelo
model_service = get_model_service()

def predict_image(image):
    """
    Función que procesa la imagen cargada por el usuario y devuelve las predicciones
    """
    if image is None:
        return {"Ropa": 0.0, "Accesorios": 0.0, "Calzado": 0.0}
    
    # Convertir imagen de Gradio a bytes para el modelo existente
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Usar el servicio de modelo existente
    prediction = model_service.predict(img_byte_arr)
    
    # Formatear los resultados para Gradio
    probabilities = prediction['probabilities']
    
    return probabilities

def create_interface():
    """Función para crear la interfaz de Gradio (facilita el mockeo en pruebas)"""
    # Buscar todas las imágenes de ejemplo disponibles
    example_images = []
    
    # Buscar imágenes por patrón
    example_files = glob.glob(str(EXAMPLES_DIR / "ejemplo_*.jpg"))
    
    print(f"Buscando imágenes de ejemplo en: {EXAMPLES_DIR}")
    
    if example_files:
        # Convertir cada archivo de ejemplo en una lista de un elemento para Gradio
        example_images = [[file] for file in example_files]
        print(f"Encontradas {len(example_files)} imágenes de ejemplo:")
        for img_file in example_files:
            print(f"  - {img_file}")
    else:
        # Verificar individualmente cada archivo
        for example_file in [EXAMPLE_ROPA, EXAMPLE_ACCESORIOS, EXAMPLE_CALZADO]:
            if example_file.exists():
                example_images.append([str(example_file)])
                print(f"Usando ejemplo: {example_file}")
    
    if not example_images:
        print("ADVERTENCIA: No se encontraron imágenes de ejemplo.")
    
    # Intentar usar un tema personalizado
    try:
        custom_theme = gr.themes.Base(
            primary_hue="purple",
            secondary_hue="blue",
            neutral_hue="gray"
        )
        theme_to_use = custom_theme
        print("Usando tema personalizado")
    except Exception as e:
        # Si falla, usar un tema simple
        print(f"Error al crear tema personalizado: {e}")
        theme_to_use = "default"
        
    return gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Sube tu imagen de moda"),
        outputs=gr.Label(num_top_classes=3, label="Resultados"),
        title="Clasificador de Moda",
        description="Sube una imagen de un producto de moda para clasificarlo como Ropa, Accesorios o Calzado.",
        theme=theme_to_use,
        examples=example_images
    )

# Crear la interfaz de Gradio
demo = create_interface()

# Iniciar la aplicación Gradio
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)