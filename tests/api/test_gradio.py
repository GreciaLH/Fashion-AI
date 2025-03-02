import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import importlib

# Agregar la ruta del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/api')))

class TestGradioInterface:
    """Tests para la interfaz de Gradio."""

    def test_predict_image_function(self, sample_pil_image):
        """Prueba que la función predict_image procesa correctamente una imagen."""
        # Este enfoque reemplaza el model_service directamente en lugar de intentar parchear la clase
        import app_gradio
        
        # Guardar la referencia original para restaurarla después
        original_service = app_gradio.model_service
        
        try:
            # Crear y configurar el mock
            mock_service = MagicMock()
            mock_service.predict.return_value = {
                'class': 'Apparel',
                'confidence': 0.8,
                'probabilities': {
                    'Apparel': 0.8,
                    'Accessories': 0.15,
                    'Footwear': 0.05
                }
            }
            
            # Reemplazar el service con nuestro mock
            app_gradio.model_service = mock_service
            
            # Llamar a la función bajo prueba
            result = app_gradio.predict_image(sample_pil_image)
            
            # Verificar resultado
            assert isinstance(result, dict)
            assert 'Apparel' in result
            assert 'Accessories' in result
            assert 'Footwear' in result
            assert result['Apparel'] == 0.8
            
            # Verificar que el mock fue llamado
            mock_service.predict.assert_called_once()
        finally:
            # Restaurar el service original
            app_gradio.model_service = original_service
    
    def test_predict_image_handles_none_input(self):
        """Prueba que predict_image maneja correctamente una entrada None."""
        import app_gradio
        
        # Guardar referencia original
        original_service = app_gradio.model_service
        
        try:
            # Crear y configurar mock
            mock_service = MagicMock()
            app_gradio.model_service = mock_service
            
            # Probar función con None
            result = app_gradio.predict_image(None)
            
            # Verificar resultado esperado
            assert isinstance(result, dict)
            assert result == {"Apparel": 0.0, "Accessories": 0.0, "Footwear": 0.0}
            
            # Verificar que el service no fue llamado
            mock_service.predict.assert_not_called()
        finally:
            # Restaurar service original
            app_gradio.model_service = original_service
    
    def test_create_interface_function(self):
        """Prueba la función create_interface directamente."""
        # Usar decorador patch para las dependencias
        with patch('app_gradio.gr') as mock_gr:
            # Configurar los mocks necesarios
            mock_gr.Image.return_value = "mocked_image_input"
            mock_gr.Label.return_value = "mocked_label_output"
            
            # Configurar el mock de Interface correctamente
            mock_interface = MagicMock()
            mock_gr.Interface = mock_interface
            
            # Importar después de configurar mocks
            import app_gradio
            
            # Llamar a la función bajo prueba
            result = app_gradio.create_interface()
            
            # Verificar que Interface fue llamado con los argumentos correctos
            mock_gr.Interface.assert_called_once()
            _, kwargs = mock_gr.Interface.call_args
            assert kwargs['fn'] == app_gradio.predict_image
            assert kwargs['inputs'] == "mocked_image_input"
            assert kwargs['outputs'] == "mocked_label_output"
            assert 'theme' in kwargs
            
            # No verificar el valor de retorno exacto, ya que es un mock
            assert result is not None