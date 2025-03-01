import pytest
import torch
import numpy as np
from PIL import Image
import io
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Asegurar que podemos importar desde src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_image():
    """Crear una imagen de muestra para pruebas"""
    # Crear una imagen RGB simple de 100x100 píxeles (roja)
    img = Image.new('RGB', (100, 100), color='red')
    
    # Convertir a bytes como si fuera cargada desde un archivo
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    
    return img_byte_arr.getvalue()

@pytest.fixture
def sample_pil_image():
    """Fixture que genera una imagen PIL para pruebas."""
    return Image.new('RGB', (224, 224), color='blue')

@pytest.fixture
def mock_tensor():
    """Fixture que genera un tensor de prueba simulando una imagen procesada."""
    return torch.rand(1, 3, 224, 224)

@pytest.fixture
def mock_model():
    """Fixture que crea un modelo mock para pruebas."""
    model = MagicMock()
    # Configurar el modelo mock para devolver un tensor de salida típico
    model.return_value = torch.tensor([[0.7, 0.2, 0.1]])
    return model

@pytest.fixture
def mock_transform():
    """Fixture que crea una transformación mock."""
    transform = MagicMock()
    transform.return_value = torch.rand(3, 224, 224)
    return transform

@pytest.fixture
def sample_prediction():
    """Fixture que devuelve una predicción de ejemplo."""
    return {
        'class': 'Apparel',
        'confidence': 0.8,
        'probabilities': {
            'Apparel': 0.8,
            'Accessories': 0.15,
            'Footwear': 0.05
        }
    }

# Fixture para pruebas de FastAPI
@pytest.fixture
def test_app():
    """Fixture que configura la aplicación FastAPI para pruebas."""
    from fastapi.testclient import TestClient
    with patch('model_service.FashionClassifierService'):
        # Importar aquí para asegurar que patch se aplique primero
        from main import app
        return TestClient(app)