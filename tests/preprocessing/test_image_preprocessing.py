import pytest
import torch
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import sys
from pathlib import Path

# Agregar la ruta correcta al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/preprocessing')))

# Importar módulos a probar
from image_preprocessing import FashionDataset, get_transforms, create_data_loaders

class TestImagePreprocessing:
    """Tests para el preprocesamiento de imágenes."""
    
    def test_get_transforms_returns_different_pipelines(self):
        """Prueba que get_transforms devuelve diferentes pipelines para train y val."""
        # Obtener transformaciones
        train_transform = get_transforms(train=True)
        val_transform = get_transforms(train=False)
        
        # Verificar que son objetos diferentes
        assert train_transform != val_transform
        
        # Verificar que son transformaciones válidas
        assert str(type(train_transform).__module__).startswith('torchvision.transforms')
        assert str(type(val_transform).__module__).startswith('torchvision.transforms')
    
    @patch('image_preprocessing.Path')
    @patch('image_preprocessing.Image.open')
    def test_fashion_dataset_getitem(self, mock_image_open, mock_path):
        """Prueba que FashionDataset.__getitem__ funciona correctamente."""
        # Crear mocks
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image
        mock_image.convert.return_value = mock_image
        
        # Mock para Path
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__.return_value = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.absolute.return_value = "/mock/path"
        
        # Crear DataFrame de muestra
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'masterCategory': ['Apparel', 'Accessories', 'Footwear']
        })
        
        # Crear mock para transform
        mock_transform = MagicMock()
        mock_transform.return_value = torch.rand(3, 224, 224)
        
        # Crear dataset
        dataset = FashionDataset("mock_dir", df, transform=mock_transform)
        
        # Probar __getitem__
        img, label = dataset[0]
        
        # Verificar resultados
        assert img is not None
        assert label in [0, 1, 2]
        mock_image_open.assert_called()
        mock_image.convert.assert_called_with('RGB')
        mock_transform.assert_called_once()
    
    @patch('image_preprocessing.Path')
    @patch('image_preprocessing.FashionDataset')
    @patch('image_preprocessing.pd.read_csv')
    @patch('image_preprocessing.os.path.exists')
    def test_create_data_loaders(self, mock_exists, mock_read_csv, mock_dataset, mock_path):
        """Prueba que create_data_loaders crea correctamente los DataLoaders."""
        # Configurar mocks
        mock_exists.return_value = True
        mock_df = MagicMock()
        mock_read_csv.return_value = mock_df
        mock_df.sample.return_value = mock_df
        mock_df.__len__.return_value = 100
        
        # Mock para los datasets
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_dataset.side_effect = [mock_train_dataset, mock_val_dataset]
        mock_train_dataset.__len__.return_value = 80
        mock_val_dataset.__len__.return_value = 20
        
        # Ejecutar función
        with patch('image_preprocessing.DataLoader') as mock_dataloader:
            mock_dataloader.return_value = MagicMock()
            train_loader, val_loader = create_data_loaders("mock_dir", "mock_csv", batch_size=32)
        
        # Verificar que se crearon correctamente
        assert train_loader is not None
        assert val_loader is not None
        
        # Verificar que se llamaron las funciones correctas
        mock_read_csv.assert_called_once()
        assert mock_dataset.call_count == 2
        assert mock_dataloader.call_count == 2