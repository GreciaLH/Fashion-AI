import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config')))
from model_config import IMAGE_SIZE

class FashionDataset(Dataset):
    """Dataset personalizado para las imágenes de moda"""
    def __init__(self, data_dir, df, transform=None):
        self.data_dir = Path(data_dir)
        print(f"Directorio de datos: {self.data_dir.absolute()}")
        
        # Filtrar primero por categoría
        df_filtered = df[df['masterCategory'].isin(['Apparel', 'Accessories', 'Footwear'])].copy()
        
        # Verificar imágenes existentes
        valid_rows = []
        total_images = len(df_filtered)
        
        for idx, row in df_filtered.iterrows():
            img_path = self.data_dir / 'images' / f"{row['id']}.jpg"
            if img_path.exists():
                valid_rows.append(row)
            
            if (len(valid_rows) + 1) % 1000 == 0:
                print(f"Verificando imágenes: {len(valid_rows)}/{total_images}")
        
        # Crear nuevo DataFrame con las filas válidas
        self.df = pd.DataFrame(valid_rows)
        print(f"\nImágenes válidas encontradas: {len(self.df)}/{total_images}")
        
        self.transform = transform
        self.class_to_idx = {
            'Apparel': 0,
            'Accessories': 1,
            'Footwear': 2
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_path = self.data_dir / 'images' / f"{row['id']}.jpg"
            
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            label = self.class_to_idx[row['masterCategory']]
            return image, label
            
        except Exception as e:
            print(f"Error procesando imagen {idx}: {e}")
            return None, None

def get_transforms(train=True):
    """Obtener transformaciones para las imágenes"""
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

def create_data_loaders(data_dir, csv_path, batch_size=32):
    """Crear DataLoaders para el dataset"""
    if not os.path.exists(csv_path):
        print(f"El archivo {csv_path} no existe.")
        return None, None

    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        print(f"Total de registros en CSV: {len(df)}")
        
        # Dividir datos antes de crear los datasets
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)
        
        print("Creando dataset de entrenamiento...")
        train_dataset = FashionDataset(data_dir, train_df, transform=get_transforms(train=True))
        
        print("\nCreando dataset de validación...")
        val_dataset = FashionDataset(data_dir, val_df, transform=get_transforms(train=False))
        
        # Crear dataloaders solo si hay datos válidos
        if len(train_dataset) > 0 and len(val_dataset) > 0:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=True
            )
            
            print(f"\nDataLoader de entrenamiento: {len(train_loader.dataset)} imágenes")
            print(f"DataLoader de validación: {len(val_loader.dataset)} imágenes")
            
            return train_loader, val_loader
        else:
            print("No hay suficientes datos válidos para crear los DataLoaders")
            return None, None
            
    except Exception as e:
        print(f"Error al crear los dataloaders: {e}")
        return None, None

if __name__ == "__main__":
    data_dir = 'data/raw'
    csv_path = 'data/raw/styles.csv'
    train_loader, val_loader = create_data_loaders(data_dir, csv_path)