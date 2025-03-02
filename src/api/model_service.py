import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import sys
import os
from pathlib import Path

# Agregar rutas necesarias
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from classifier import FashionClassifier

class FashionClassifierService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.transform = self.get_transforms()
        # Cambiar los nombres de las clases al español
        self.classes = ['Ropa', 'Accesorios', 'Calzado']
        # Mapeo de nombres de clases en inglés a español para compatibilidad
        self.class_mapping = {
            'Apparel': 'Ropa',
            'Accessories': 'Accesorios',
            'Footwear': 'Calzado'
        }
    
    def load_model(self):
        model = FashionClassifier(num_classes=3)
        model_path = Path('models/saved_models/best_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def get_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_bytes):
        # Convertir bytes a imagen
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocesar imagen
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Realizar predicción
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'class': self.classes[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(self.classes, probabilities[0])
            }
        }