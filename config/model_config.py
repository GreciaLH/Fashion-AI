# Configuración del preprocesamiento
IMAGE_SIZE = 224  # Tamaño estándar para modelos preentrenados
BATCH_SIZE = 32
NUM_WORKERS = 4

# Categorías principales
CATEGORIES = ['Apparel', 'Accessories', 'Footwear']

# División del dataset
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Configuración de aumento de datos
DATA_AUGMENTATION = {
    'horizontal_flip': True,
    'rotation_range': 20,
    'brightness_range': (0.8, 1.2),
    'zoom_range': 0.2
}