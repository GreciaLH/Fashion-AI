import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Agregar rutas necesarias
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../preprocessing')))

from classifier import FashionClassifier
from image_preprocessing import create_data_loaders

def load_trained_model(model_path, device):
    """Cargar el modelo entrenado"""
    model = FashionClassifier(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, val_loader, device):
    """Evaluar el modelo y obtener predicciones y etiquetas verdaderas"""
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluando modelo'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = outputs.max(1)
            
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predictions.cpu().numpy())
    
    return np.array(true_labels), np.array(pred_labels)

def plot_confusion_matrix(true_labels, pred_labels, class_names):
    """Visualizar matriz de confusión"""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Predicción')
    plt.tight_layout()
    
    # Guardar la figura
    save_dir = Path('reports/figures')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'confusion_matrix.png')
    plt.close()

def analyze_errors(true_labels, pred_labels, val_loader, class_names):
    """Analizar ejemplos de predicciones incorrectas"""
    errors = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            for j, (image, label) in enumerate(zip(images, labels)):
                idx = i * val_loader.batch_size + j
                if idx < len(true_labels) and true_labels[idx] != pred_labels[idx]:
                    errors.append({
                        'image': image,
                        'true': class_names[label],
                        'pred': class_names[pred_labels[idx]]
                    })
    return errors

def main():
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Cargar datos
    data_dir = Path('data/raw')
    csv_path = data_dir / 'styles.csv'
    _, val_loader = create_data_loaders(data_dir, csv_path, batch_size=32)
    
    # Cargar modelo
    model_path = Path('models/saved_models/best_model.pth')
    model = load_trained_model(model_path, device)
    
    # Nombres de las clases
    class_names = ['Apparel', 'Accessories', 'Footwear']
    
    # Evaluar modelo
    print("\nEvaluando modelo...")
    true_labels, pred_labels = evaluate_model(model, val_loader, device)
    
    # Generar y mostrar métricas
    print("\nReporte de Clasificación:")
    print(classification_report(true_labels, pred_labels, target_names=class_names))
    
    # Visualizar matriz de confusión
    print("\nGenerando matriz de confusión...")
    plot_confusion_matrix(true_labels, pred_labels, class_names)
    print("Matriz de confusión guardada en 'reports/figures/confusion_matrix.png'")
    
    # Analizar errores
    print("\nAnalizando errores de clasificación...")
    errors = analyze_errors(true_labels, pred_labels, val_loader, class_names)
    print(f"Número total de errores encontrados: {len(errors)}")
    
    # Guardar métricas en un archivo
    metrics_dir = Path('reports/metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_dir / 'evaluation_results.txt', 'w') as f:
        f.write("Resultados de la Evaluación del Modelo\n")
        f.write("=====================================\n\n")
        f.write(classification_report(true_labels, pred_labels, target_names=class_names))
        f.write(f"\nNúmero total de errores: {len(errors)}")

if __name__ == "__main__":
    main()