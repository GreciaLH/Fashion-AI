import pytest
import torch
from unittest.mock import patch, MagicMock
import sys
import os
import inspect
from pathlib import Path

# Agregar rutas necesarias al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/models')))

# Ahora podemos importar el módulo classifier
from classifier import FashionClassifier

class TestFashionClassifier:
    """Tests para el modelo FashionClassifier."""
    
    def test_init_creates_correct_model_structure(self):
        """Prueba que el modelo se inicializa con la estructura correcta."""
        # Crear modelo
        model = FashionClassifier(num_classes=3)
        
        # Verificar que es una instancia de nn.Module
        assert isinstance(model, torch.nn.Module)
        
        # Verificar que la última capa tiene las salidas correctas
        assert model.model.fc.out_features == 3
    
    def test_forward_returns_correct_output_shape(self):
        """Prueba que el método forward devuelve un tensor con la forma correcta."""
        # Crear modelo
        model = FashionClassifier(num_classes=3)
        
        # Crear tensor de entrada de muestra (batch_size=2)
        x = torch.rand(2, 3, 224, 224)
        
        # Realizar forward pass
        with torch.no_grad():
            output = model(x)
        
        # Verificar forma de salida
        assert output.shape == (2, 3)  # (batch_size, num_classes)
    
    @patch('classifier.models.resnet18')
    def test_model_uses_pretrained_weights(self, mock_resnet):
        """Prueba que el modelo usa pesos preentrenados de ResNet18."""
        # Crear modelo
        _ = FashionClassifier(num_classes=3)
        
        # Verificar que se llamó a resnet18 con weights
        mock_resnet.assert_called_once()
        args, kwargs = mock_resnet.call_args
        assert 'weights' in kwargs
    
    def test_train_model_uses_correct_loss_and_optimizer(self):
        """Prueba que la función train_model usa la pérdida y optimizador correctos."""
        # Importar directamente la función para analizar su código
        from classifier import train_model
        
        # Verificar que la función contiene los componentes correctos (inspección de código)
        source = inspect.getsource(train_model)
        
        # Verificar que se usa CrossEntropyLoss
        assert "criterion = nn.CrossEntropyLoss()" in source
        
        # Verificar que se usa Adam con lr=0.001
        assert "optimizer = Adam(model.parameters(), lr=0.001)" in source
        
        # Verificar que llama a model.train() y model.eval()
        assert "model.train()" in source
        assert "model.eval()" in source