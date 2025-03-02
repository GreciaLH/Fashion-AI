import pytest
from unittest.mock import patch, MagicMock
import torch
import io
from PIL import Image
from fastapi.testclient import TestClient
import sys
import os

# Agregar rutas necesarias al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/api')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/models')))

class TestIntegration:
    """Tests de integración para el sistema completo."""
    
    def test_end_to_end_prediction_flow(self, test_app):
        """Prueba el flujo completo desde la solicitud hasta la predicción."""
        # Desempaquetar cliente y mock_service del fixture test_app
        client, mock_service = test_app
        
        # Configurar el mock_service para que devuelva una predicción específica
        mock_service.predict.return_value = {
            'class': 'Apparel',
            'confidence': 0.8,
            'probabilities': {
                'Apparel': 0.8,
                'Accessories': 0.1,
                'Footwear': 0.1
            }
        }
        
        # Crear una imagen de prueba
        img = Image.new('RGB', (224, 224), color='blue')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Enviar solicitud al endpoint
        files = {'file': ('test_image.jpg', img_byte_arr, 'image/jpeg')}
        response = client.post("/predict", files=files)
        
        # Verificar que la respuesta es correcta
        assert response.status_code == 200
        json_data = response.json()
        
        # Verificar estructura de la respuesta
        assert 'class' in json_data
        assert 'confidence' in json_data
        assert 'probabilities' in json_data
        
        # Verificar que las probabilidades suman aproximadamente 1
        probs = json_data['probabilities']
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 0.01
    
    @patch('model_service.FashionClassifierService.load_model')
    def test_model_service_transform_integration(self, mock_load_model):
        """Prueba la integración entre transformaciones y el servicio del modelo."""
        # Importamos después de aplicar el patch
        from model_service import FashionClassifierService
        
        # Configurar mock del modelo
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Configurar comportamiento del mock del modelo
        def mock_model_output(*args, **kwargs):
            return torch.tensor([[0.6, 0.3, 0.1]])
        mock_model.side_effect = mock_model_output
        
        # Crear servicio con el modelo mock
        service = FashionClassifierService()
        
        # Sobrescribir el modelo para asegurarnos que usa el mock
        service.model = mock_model
        
        # Crear imagen de prueba
        img = Image.new('RGB', (224, 224), color='green')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        # Realizar predicción
        result = service.predict(img_bytes)
        
        # Verificar formato correcto
        assert isinstance(result, dict)
        assert 'class' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        
        # Verificar que las clases son correctas
        assert result['class'] in ['Apparel', 'Accessories', 'Footwear']