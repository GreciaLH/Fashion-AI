import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from tqdm import tqdm
import sys
import os
from pathlib import Path

# Agregar rutas necesarias
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../preprocessing')))
from image_preprocessing import create_data_loaders

class FashionClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(FashionClassifier, self).__init__()
        # Usar ResNet18 con pesos preentrenados
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modificar la última capa para nuestras 3 clases
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, num_epochs=10, device='cpu'):
    # Definir pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Crear directorio para guardar modelos si no existe
    model_dir = Path('models/saved_models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Variables para seguimiento
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Modo entrenamiento
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        # Barra de progreso para entrenamiento
        train_pbar = tqdm(train_loader, desc=f'Training')
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calcular accuracy
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            train_loss += loss.item()
            
            # Actualizar barra de progreso
            train_pbar.set_postfix({
                'loss': train_loss/(batch_idx+1),
                'acc': 100.*train_correct/train_total
            })
        
        # Modo evaluación
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Validation')
            for batch_idx, (images, labels) in enumerate(val_pbar):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': val_loss/(batch_idx+1),
                    'acc': 100.*val_correct/val_total
                })
        
        # Calcular métricas de la época
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        
        # Guardar resultados
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        print(f'\nEpoch Summary:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        
        # Guardar mejor modelo
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(model.state_dict(), model_dir / 'best_model.pth')
            print(f'Nuevo mejor modelo guardado con accuracy: {best_val_accuracy:.2f}%')

if __name__ == "__main__":
    # Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Crear data loaders
    data_dir = Path('data/raw')
    csv_path = data_dir / 'styles.csv'
    train_loader, val_loader = create_data_loaders(data_dir, csv_path, batch_size=32)
    
    if train_loader is not None and val_loader is not None:
        # Crear y entrenar modelo
        model = FashionClassifier(num_classes=3)
        model = model.to(device)
        
        print("\nIniciando entrenamiento...")
        train_model(model, train_loader, val_loader, num_epochs=10, device=device)
        print("\nEntrenamiento completado!")