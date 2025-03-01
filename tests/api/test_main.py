import pytest
from unittest.mock import patch, MagicMock
import io
from PIL import Image
import sys
import os

# Agregar la ruta del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/api')))

# Vamos a probar los endpoints de la API
class TestFashionAPI:
    """Tests para los endpoints de la API de clasificación de moda."""
    
    def test_predict_endpoint_returns_correct_response(self, test_app):
        """Prueba que el endpoint /predict devuelve la respuesta correcta."""
        # Desempaquetar los valores del fixture
        client, mock_service = test_app
        
        # Configurar el mock para devolver una predicción predefinida
        mock_service.predict.return_value = {
            'class': 'Apparel',
            'confidence': 0.8,
            'probabilities': {
                'Apparel': 0.8,
                'Accessories': 0.15,
                'Footwear': 0.05
            }
        }
        
        # Crear una imagen de prueba
        img = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Enviar solicitud al endpoint
        files = {'file': ('test_image.jpg', img_byte_arr, 'image/jpeg')}
        response = client.post("/predict", files=files)
        
        # Verificar respuesta
        assert response.status_code == 200
        json_response = response.json()
        assert 'class' in json_response
        assert 'confidence' in json_response
        assert 'probabilities' in json_response
        assert json_response['class'] == 'Apparel'
    
    def test_predict_endpoint_handles_invalid_image(self, test_app):
        """Prueba que el endpoint /predict maneja correctamente imágenes inválidas."""
        # Desempaquetar los valores del fixture
        client, _ = test_app
        
        # Enviar solicitud con un archivo que no es una imagen
        files = {'file': ('test.txt', b'This is not an image', 'text/plain')}
        response = client.post("/predict", files=files)
        
        # Verificar respuesta de error
        assert response.status_code == 400
        assert "El archivo debe ser una imagen" in response.text
    
    def test_predict_endpoint_handles_model_errors(self, test_app):
        """Prueba que el endpoint /predict maneja errores del modelo correctamente."""
        # Desempaquetar los valores del fixture
        client, mock_service = test_app
        
        # Configurar el mock para lanzar una excepción
        mock_service.predict.side_effect = Exception("Error de modelo simulado")
        
        # Crear una imagen de prueba
        img = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Enviar solicitud al endpoint
        files = {'file': ('test_image.jpg', img_byte_arr, 'image/jpeg')}
        response = client.post("/predict", files=files)
        
        # Verificar respuesta de error
        assert response.status_code == 500
        assert "Error al procesar la imagen" in response.text
