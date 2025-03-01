import pytest
import torch
from unittest.mock import patch, MagicMock
import io
from PIL import Image
import sys
import os

# Agregar la ruta del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/api')))

# Ahora importar el módulo
from model_service import FashionClassifierService

class TestFashionClassifierService:
    """Tests para el servicio del clasificador de moda."""
    
    @patch('model_service.FashionClassifier')
    @patch('model_service.torch.load')
    def test_init_loads_model_correctly(self, mock_torch_load, mock_classifier_class):
        """Prueba que el servicio inicializa y carga el modelo correctamente."""
        # Configurar mocks
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model
        mock_torch_load.return_value = {}
        
        # Inicializar servicio
        service = FashionClassifierService()
        
        # Verificar que el modelo se inicializó correctamente
        mock_classifier_class.assert_called_once()
        mock_torch_load.assert_called_once()
        mock_model.load_state_dict.assert_called_once()
        mock_model.eval.assert_called_once()
        
        # Verificar que las clases se configuraron correctamente
        assert service.classes == ['Apparel', 'Accessories', 'Footwear']
    
    @patch('model_service.FashionClassifier')
    @patch('model_service.torch.load')
    def test_get_transforms_returns_correct_pipeline(self, mock_torch_load, mock_classifier_class):
        """Prueba que el método get_transforms devuelve una pipeline de transformación válida."""
        # Configurar mocks
        mock_classifier_class.return_value = MagicMock()
        
        # Inicializar servicio
        service = FashionClassifierService()
        
        # Obtener transformaciones
        transform = service.get_transforms()
        
        # Verificar que las transformaciones son correctas (comprobamos el tipo)
        assert str(type(transform).__module__).startswith('torchvision.transforms')
    
    @patch('model_service.FashionClassifier')
    @patch('model_service.torch.load')
    def test_predict_returns_correct_format(self, mock_torch_load, mock_classifier_class, sample_image):
        """Prueba que el método predict devuelve resultados en el formato correcto."""
        # Configurar mocks
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model
        
        # Configurar el modelo para devolver una predicción específica
        mock_model.return_value = torch.tensor([[0.7, 0.2, 0.1]])
        
        # Inicializar servicio
        service = FashionClassifierService()
        
        # Reemplazar transform con un mock para evitar errores
        service.transform = MagicMock()
        service.transform.return_value = torch.rand(3, 224, 224)
        
        # Realizar predicción
        result = service.predict(sample_image)
        
        # Verificar formato y contenido
        assert isinstance(result, dict)
        assert 'class' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert result['class'] in service.classes
        assert isinstance(result['confidence'], float)
        assert isinstance(result['probabilities'], dict)
        assert all(cls in result['probabilities'] for cls in service.classes)
    
    @patch('model_service.FashionClassifier')
    @patch('model_service.torch.load')
    def test_predict_handles_different_image_formats(self, mock_torch_load, mock_classifier_class):
        """Prueba que predict puede manejar diferentes formatos de imagen."""
        # Configurar mocks
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model
        mock_model.return_value = torch.tensor([[0.7, 0.2, 0.1]])
        
        # Inicializar servicio
        service = FashionClassifierService()
        service.transform = MagicMock()
        service.transform.return_value = torch.rand(3, 224, 224)
        
        # Probar con diferentes formatos
        for color in ['red', 'green', 'blue']:
            # Crear imagen RGB
            img = Image.new('RGB', (224, 224), color=color)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            
            # Realizar predicción
            result = service.predict(img_byte_arr.getvalue())
            
            # Verificar resultado
            assert isinstance(result, dict)
            assert 'class' in result