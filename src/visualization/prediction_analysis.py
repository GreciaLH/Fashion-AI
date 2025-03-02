import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os
from torchvision import transforms
from PIL import Image
import random
from tqdm import tqdm

# Agregar rutas necesarias
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../preprocessing')))

from classifier import FashionClassifier
from image_preprocessing import create_data_loaders

class PredictionVisualizer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        # Cambiar los nombres de las clases al español
        self.classes = ['Ropa', 'Accesorios', 'Calzado']
        
    def load_model(self, model_path):
        model = FashionClassifier(num_classes=3)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def visualize_predictions(self, dataloader, num_samples=10):
        """Visualizar predicciones correctas e incorrectas"""
        correct_predictions = []
        incorrect_predictions = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Analizando predicciones"):
                outputs = self.model(images)
                _, predictions = outputs.max(1)
                
                # Guardar índices de predicciones correctas e incorrectas
                for idx, (pred, label) in enumerate(zip(predictions, labels)):
                    if len(correct_predictions) < num_samples and pred == label:
                        correct_predictions.append({
                            'image': images[idx],
                            'pred': pred.item(),
                            'true': label.item()
                        })
                    elif len(incorrect_predictions) < num_samples and pred != label:
                        incorrect_predictions.append({
                            'image': images[idx],
                            'pred': pred.item(),
                            'true': label.item()
                        })
                
                if (len(correct_predictions) >= num_samples and 
                    len(incorrect_predictions) >= num_samples):
                    break
        
        return correct_predictions, incorrect_predictions

    def plot_predictions(self, predictions, title):
        """Visualizar un conjunto de predicciones"""
        n = len(predictions)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(title)
        
        for idx, pred in enumerate(predictions):
            ax = axes[idx//5, idx%5]
            
            # Convertir tensor a imagen
            img = pred['image'].cpu().permute(1, 2, 0)
            img = self.denormalize_image(img)
            
            ax.imshow(img)
            ax.set_title(f'Pred: {self.classes[pred["pred"]]}\nTrue: {self.classes[pred["true"]]}')
            ax.axis('off')
        
        plt.tight_layout()
        return fig

    @staticmethod
    def denormalize_image(img):
        """Desnormalizar imagen para visualización"""
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = img * std + mean
        img = img.clip(0, 1)
        return img.numpy()

def main():
    # Configurar rutas
    data_dir = Path('data/raw')
    csv_path = data_dir / 'styles.csv'
    model_path = Path('models/saved_models/best_model.pth')
    
    # Crear directorio para guardar visualizaciones
    save_dir = Path('reports/visualizations')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos y modelo
    _, val_loader = create_data_loaders(data_dir, csv_path, batch_size=32)
    visualizer = PredictionVisualizer(model_path)
    
    # Obtener y visualizar predicciones
    print("\nAnalizando predicciones...")
    correct_preds, incorrect_preds = visualizer.visualize_predictions(val_loader)
    
    # Visualizar predicciones correctas
    print("\nGenerando visualización de predicciones correctas...")
    fig_correct = visualizer.plot_predictions(correct_preds, 'Predicciones Correctas')
    fig_correct.savefig(save_dir / 'correct_predictions.png')
    
    # Visualizar predicciones incorrectas
    print("\nGenerando visualización de predicciones incorrectas...")
    fig_incorrect = visualizer.plot_predictions(incorrect_preds, 'Predicciones Incorrectas')
    fig_incorrect.savefig(save_dir / 'incorrect_predictions.png')
    
    print("\nVisualizaciones guardadas en:")
    print(f"- {save_dir / 'correct_predictions.png'}")
    print(f"- {save_dir / 'incorrect_predictions.png'}")

if __name__ == "__main__":
    main()