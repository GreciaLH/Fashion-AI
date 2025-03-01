import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import io
from PIL import Image

# Pruebas para la interfaz Gradio
class TestGradioInterface:
    """Tests para la interfaz de Gradio."""

    @patch('app_gradio.FashionClassifierService')
    def test_predict_image_function(self, mock_service_class, sample_pil_image):
        """Prueba que la función predict_image procesa correctamente una imagen."""
        # Configurar mock
        mock_service = mock_service_class.return_value
        mock_service.predict.return_value = {
            'class': 'Apparel',
            'confidence': 0.8,
            'probabilities': {
                'Apparel': 0.8,
                'Accessories': 0.15,
                'Footwear': 0.05
            }
        }
        
        # Importar función a probar después de configurar el mock
        from app_gradio import predict_image
        
        # Probar función con una imagen de muestra
        result = predict_image(sample_pil_image)
        
        # Verificar resultado
        assert isinstance(result, dict)
        assert 'Apparel' in result
        assert 'Accessories' in result
        assert 'Footwear' in result
        assert result['Apparel'] == 0.8
    
    @patch('app_gradio.FashionClassifierService')
    def test_predict_image_handles_none_input(self, mock_service_class):
        """Prueba que predict_image maneja correctamente una entrada None."""
        # Importar función a probar después de configurar el mock
        from app_gradio import predict_image
        
        # Probar función con None
        result = predict_image(None)
        
        # Verificar que devuelve valores por defecto
        assert isinstance(result, dict)
        assert result == {"Apparel": 0.0, "Accessories": 0.0, "Footwear": 0.0}
    
    @patch('app_gradio.gr.Interface')
    @patch('app_gradio.FashionClassifierService')
    def test_gradio_interface_creation(self, mock_service_class, mock_interface):
        """Prueba que la interfaz Gradio se crea correctamente."""
        # Importar módulo después de configurar los mocks
        import app_gradio
        
        # Verificar que se creó la interfaz
        mock_interface.assert_called_once()
        
        # Verificar argumentos básicos
        args, kwargs = mock_interface.call_args
        assert 'fn' in kwargs
        assert 'inputs' in kwargs
        assert 'outputs' in kwargs
        assert 'theme' in kwargs