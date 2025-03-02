# Fashion AI - Deep Learning para Clasificación de Moda


Un sistema de clasificación de imágenes de moda utilizando deep learning con ResNet18 que clasifica productos en tres categorías: Ropa (Apparel), Accesorios (Accessories) y Calzado (Footwear).

## Tabla de Contenidos

- [Visión General](#visión-general)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Características Principales](#características-principales)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Arquitectura del Modelo](#arquitectura-del-modelo)
- [Preprocesamiento de Datos](#preprocesamiento-de-datos)
- [Entrenamiento del Modelo](#entrenamiento-del-modelo)
- [Evaluación de Resultados](#evaluación-de-resultados)
- [API y Interfaz de Usuario](#api-y-interfaz-de-usuario)
- [Pruebas Unitarias](#pruebas-unitarias)
- [Instalación y Uso](#instalación-y-uso)
- [Resultados y Métricas](#resultados-y-métricas)
- [Futuras Mejoras](#futuras-mejoras)
- [Contribuciones](#contribuciones)

## Visión General

Este proyecto implementa un clasificador de imágenes de moda utilizando transfer learning con ResNet18. El sistema está diseñado para analizar imágenes de productos y clasificarlas en tres categorías principales:

- **Apparel** (Ropa)
- **Accessories** (Accesorios)
- **Footwear** (Calzado)

La solución incluye una API RESTful construida con FastAPI y una interfaz gráfica moderna implementada con Gradio, permitiendo a los usuarios interactuar fácilmente con el modelo.

## Estructura del Proyecto

```
fashion-classifier/
├── config/
│   └── model_config.py       # Configuración global del modelo
├── data/
│   └── raw/
│       ├── images/           # Imágenes de entrenamiento
│       └── styles.csv        # Metadata de productos
├── models/
│   └── saved_models/
│       └── best_model.pth    # Modelo entrenado guardado
├── notebooks/                # Jupyter notebooks de análisis
├── reports/
│   ├── figures/              # Gráficos y visualizaciones
│   ├── metrics/              # Reportes de evaluación
│   └── visualizations/       # Visualizaciones de predicciones
├── src/
│   ├── api/
│   │   ├── app_gradio.py     # Interfaz Gradio
│   │   ├── main.py           # API FastAPI
│   │   └── model_service.py  # Servicio del modelo
│   ├── evaluation/
│   │   └── model_evaluation.py  # Evaluación del modelo
│   ├── models/
│   │   └── classifier.py     # Definición del modelo
│   ├── preprocessing/
│   │   └── image_preprocessing.py  # Preprocesamiento de datos
│   └── visualization/
│       └── prediction_analysis.py  # Análisis de predicciones
├── tests/                    # Tests unitarios
├── requirements.txt          # Dependencias
└── README.md
```

## Características Principales

- **Clasificación precisa** (99% de precisión) en tres categorías de productos de moda
- **Transfer Learning** utilizando ResNet18 preentrenado
- **API RESTful** para integración con otros sistemas
- **Interfaz gráfica** con Gradio
- **Pipeline completo de preprocesamiento** de imágenes
- **Pruebas unitarias exhaustivas** para garantizar calidad del código
- **Evaluación extensiva** del rendimiento del modelo

## Tecnologías Utilizadas

- **PyTorch**: Framework de deep learning para implementación y entrenamiento del modelo
- **FastAPI**: Para la creación de la API RESTful
- **Gradio**: Para la interfaz gráfica interactiva
- **Pandas**: Para manipulación de datos y análisis
- **Pillow**: Para procesamiento de imágenes
- **scikit-learn**: Para métricas de evaluación
- **matplotlib/seaborn**: Para visualizaciones
- **pytest**: Para pruebas unitarias

## Arquitectura del Modelo

El clasificador se basa en una arquitectura ResNet18 modificada:

```python
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
```

La arquitectura utiliza transfer learning, aprovechando el poder de representación de características aprendidas en ImageNet y adaptándolas específicamente para la clasificación de productos de moda.

## Preprocesamiento de Datos

El pipeline de preprocesamiento incluye:

- Carga de datos desde CSV y verificación de imágenes existentes
- Redimensionamiento de imágenes a 224x224 píxeles
- Normalización usando los parámetros estándar de ImageNet
- Data augmentation para el conjunto de entrenamiento:
  - Volteo horizontal aleatorio
  - Rotación aleatoria
  - Ajustes de brillo y contraste

```python
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
```

## Entrenamiento del Modelo

El proceso de entrenamiento utiliza:

- Optimizador Adam con tasa de aprendizaje de 0.001
- Función de pérdida CrossEntropyLoss
- Early stopping guardando el mejor modelo según la precisión de validación
- Monitoreo de métricas de entrenamiento y validación
- Dropout para prevenir el sobreajuste

## Evaluación de Resultados

![Matriz de Confusión](\reports\figures\confusion_matrix.png)

El modelo alcanza métricas impresionantes:
- **Precisión global**: 99%
- **Recall**: 99%
- **F1-Score**: 99%

```
Resultados de la Evaluación del Modelo
=====================================

              precision    recall  f1-score   support

     Apparel       0.99      1.00      0.99      4325
 Accessories       0.99      0.99      0.99      2165
    Footwear       1.00      1.00      1.00      1894

    accuracy                           0.99      8384
   macro avg       0.99      0.99      0.99      8384
weighted avg       0.99      0.99      0.99      8384

Número total de errores: 53
```

## API e Interfaz de Usuario

### API FastAPI

El sistema expone un endpoint para clasificación de imágenes:

```
POST /predict
```

Ejemplo de respuesta:
```json
{
  "class": "Apparel",
  "confidence": 0.92,
  "probabilities": {
    "Apparel": 0.92,
    "Accessories": 0.05,
    "Footwear": 0.03
  }
}
```

### Interfaz Gradio

La interfaz de usuario proporciona una experiencia interactiva para la clasificación de imágenes:

![Interfaz Gradio](\reports\screenshots\gradio_interface.png)

- Subida intuitiva de imágenes
- Visualización de resultados con barras de confianza
- Ejemplos pre-cargados para demostración rápida

## Pruebas Unitarias

El proyecto incluye pruebas unitarias exhaustivas para todos los componentes:

- Tests del modelo y clasificador
- Tests del servicio de predicción
- Tests de preprocesamiento de imágenes
- Tests de API y endpoints
- Tests de la interfaz Gradio
- Tests de integración

Para ejecutar las pruebas:

```bash
pip install pytest pytest-cov httpx
pytest
```

## Instalación y Uso

### Requisitos

- Python 3.8+
- PyTorch 2.0+
- FastAPI 0.99+
- Gradio 3.35+

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/GreciaLH/Fashion-AI.git
cd Fashion-AI

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución

```bash

python src/api/app_gradio.py
```

## Resultados y Métricas

El modelo fue evaluado en un conjunto de datos de 8,384 imágenes y mostró un buen rendimiento:

| Categoría    | Precisión | Recall | F1-Score | Soporte |
|--------------|-----------|--------|----------|---------|
| Apparel      | 0.99      | 1.00   | 0.99     | 4,325   |
| Accessories  | 0.99      | 0.99   | 0.99     | 2,165   |
| Footwear     | 1.00      | 1.00   | 1.00     | 1,894   |
| **Promedio** | **0.99**  | **0.99**| **0.99**| **8,384**|

## Futuras Mejoras

- Implementar clasificación de subcategorías más específicas
- Añadir detección de atributos (color, estilo, temporada)
- Integrar recomendaciones de productos similares
- Mejorar la interfaz con más opciones de análisis visual

## Contribuciones

Las contribuciones son bienvenidas. Por favor, siga estos pasos:

1. Fork el repositorio
2. Cree una rama para su característica (`git checkout -b feature/amazing-feature`)
3. Commit sus cambios (`git commit -m 'Add some amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abra un Pull Request


---

**Nota**: Este proyecto fue creado con fines educativos y de demostración. Las imágenes utilizadas pertenecen a sus respectivos propietarios.