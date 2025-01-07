import torch
import torchvision
import pandas as pd
import numpy as np
from PIL import Image

def verify_installations():
    """Verificar que todas las instalaciones funcionan correctamente"""
    # Verificar versiones
    print("Verificando instalaciones:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")
    print(f"Pandas Version: {pd.__version__}")
    print(f"Numpy Version: {np.__version__}")
    
    # Verificar dispositivo disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDispositivo disponible: {device}")

    return device

if __name__ == "__main__":
    device = verify_installations()