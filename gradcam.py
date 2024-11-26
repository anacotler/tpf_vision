import torch
import torchvision.transforms as transforms
from captum.attr import LayerGradCam, LayerAttribution
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import AgeEstimationModel
from config import config
import os

# Función para visualizar las atribuciones
def visualize_attr(attr, original_image, alpha=0.6, cmap='jet'):
    """
    Superpone un mapa de calor sobre la imagen original.
    Args:
        attr: Atribuciones generadas (numpy array 2D).
        original_image: Imagen original (numpy array HxWxC en rango [0, 1]).
        alpha: Transparencia para superposición.
        cmap: Mapa de color para el heatmap.
    """
    plt.figure(figsize=(8, 8))
    
    # Mostrar imagen original
    plt.imshow(original_image)
    
    # Superponer mapa de calor
    plt.imshow(attr, cmap=cmap, alpha=alpha)
    
    # Eliminar ejes
    plt.axis('off')
    plt.show()


# Cargar el modelo y pesos
checkpoint_files = '/home/anacotler/Documents/vision_artificial/tpf_vision/checkpoints/epoch-14-loss_valid-13.5.pt'
model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name=config['model_name'], pretrain_weights='IMAGENET1K_V2').to(config['device'])
model.load_state_dict(torch.load(checkpoint_files, map_location=config['device']))
model.eval()  # Poner el modelo en modo evaluación

# Cargar la imagen
image_path = '/home/anacotler/Documents/vision_artificial/tpf_vision/utkcropped/1_0_0_20161219140623097.jpg.chip.jpg'
image = Image.open(image_path).convert('RGB')

# Preprocesar la imagen
transform = transforms.Compose([
    transforms.Resize((config['img_width'], config['img_height'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['mean'], std=config['std'])
])
input_data = transform(image).unsqueeze(0).to(config['device'])

# Capa objetivo para Grad-CAM (última capa convolucional del modelo)
target_layer = model.model.layer4[0]

# Crear objeto LayerGradCam
layer_gc = LayerGradCam(model, target_layer)

# Calcular las atribuciones
attr = layer_gc.attribute(input_data, target=0)

# Preparar las atribuciones para visualización
attr_upsampled = LayerAttribution.interpolate(attr, (config['img_width'], config['img_height'])).squeeze(0).detach().cpu().numpy()

# Convertir la imagen original a NumPy (denormalizar si es necesario)
original_image = np.array(image) / 255.0  # Normalizar a rango [0, 1]

# Asegurarse de que las dimensiones de las atribuciones sean correctas
if len(attr_upsampled.shape) == 3 and attr_upsampled.shape[0] == 1:
    attr_upsampled = attr_upsampled.squeeze(0)  # Remover el canal extra

# Visualizar las atribuciones
visualize_attr(attr_upsampled, original_image, alpha=0.6, cmap='jet')


# from captum.attr import visualize_image_attr
# _ = visualize_image_attr(attr_upsampled, original_image, method='blended_heat_map', sign='all', show_colorbar=True, title='LayerGradCam')

# # Save the visualization
# plt.savefig('gradcam.png')

# # Display the visualization
# plt.show()

# # Prepara el original y la atribución
# original_image = np.array(image) / 255.0  # Escala imagen a rango [0, 1]
# attr_upsampled = LayerAttribution.interpolate(attr, (config['img_width'], config['img_height'])).squeeze(0).cpu().numpy()

# # Normaliza las atribuciones
# attr_upsampled = (attr_upsampled - np.min(attr_upsampled)) / (np.max(attr_upsampled) - np.min(attr_upsampled))

# # Visualiza
# visualize_attr(attr_upsampled, original_image)



