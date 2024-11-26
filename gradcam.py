'''import torch
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


'''

'''
import torch
import torchvision.transforms as transforms
from captum.attr import LayerGradCam, LayerAttribution
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import AgeEstimationModel
from config import config

# Función para visualizar las atribuciones
def visualize_attr(attr, original_image, alpha=0.6, cmap='jet', title=None, save_path=None):
    """
    Superpone un mapa de calor sobre la imagen original.
    Args:
        attr: Atribuciones generadas (numpy array 2D).
        original_image: Imagen original (numpy array HxWxC en rango [0, 1]).
        alpha: Transparencia para superposición.
        cmap: Mapa de color para el heatmap.
        title: Título de la imagen.
        save_path: Ruta donde guardar la imagen.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.imshow(attr, cmap=cmap, alpha=alpha)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=16)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Cargar el modelo y pesos
checkpoint_files = '/home/anacotler/Documents/vision_artificial/tpf_vision/checkpoints/epoch-14-loss_valid-13.5.pt'
model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name=config['model_name'], pretrain_weights='IMAGENET1K_V2').to(config['device'])
model.load_state_dict(torch.load(checkpoint_files, map_location=config['device']))
model.eval()

# Confirmar que el modelo se cargó correctamente
print("Modelo cargado correctamente")

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

# Convertir la imagen original a NumPy (denormalizar si es necesario)
original_image = np.array(image) / 255.0  # Normalizar a rango [0, 1]

# Iterar sobre las capas convolucionales de ResNet50
for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
    layer = getattr(model.model, layer_name)  # Obtener la capa por su nombre
    print(f"Procesando {layer_name}")  # Confirmar que se procesa la capa
    for sub_layer_idx, sub_layer in enumerate(layer):
        print(f"Sub capa {sub_layer_idx}: {type(sub_layer)}")  # Verificar subcapas
        if isinstance(sub_layer, torch.nn.Module):
            for block_idx, block in enumerate(sub_layer.children()):
                if isinstance(block, torch.nn.Conv2d):  # Verificar si es una capa convolucional
                    print(f"Generando atribuciones para {layer_name} - Bloque {sub_layer_idx}-{block_idx}")

                    # Crear objeto LayerGradCam
                    layer_gc = LayerGradCam(model, block)

                    # Calcular las atribuciones
                    attr = layer_gc.attribute(input_data, target=0)

                    # Preparar las atribuciones para visualización
                    attr_upsampled = LayerAttribution.interpolate(attr, (config['img_width'], config['img_height'])).squeeze(0).detach().cpu().numpy()

                    # Asegurarse de que las dimensiones sean correctas
                    if len(attr_upsampled.shape) == 3 and attr_upsampled.shape[0] == 1:
                        attr_upsampled = attr_upsampled.squeeze(0)

                    # Visualizar las atribuciones
                    title = f'{layer_name} - Block {sub_layer_idx}-{block_idx}'
                    save_path = f'output_{layer_name}_{sub_layer_idx}_{block_idx}.png'  # Guardar cada imagen
                    visualize_attr(attr_upsampled, original_image, alpha=0.6, cmap='jet', title=title, save_path=save_path)

print("Atribuciones generadas y visualizadas correctamente.")

'''


import torch
import torchvision.transforms as transforms
from captum.attr import LayerGradCam, LayerAttribution
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import AgeEstimationModel
from config import config

# Función para crear una grilla de visualizaciones
def plot_attributions_grid(attributions, original_image, cols=4, cmap='jet'):
    """
    Muestra las atribuciones en una sola imagen como una grilla.
    Args:
        attributions: Lista de tuples (atribución, título).
        original_image: Imagen original (numpy array HxWxC en rango [0, 1]).
        cols: Número de columnas en la grilla.
        cmap: Mapa de color para el heatmap.
    """
    rows = (len(attributions) + cols - 1) // cols  # Calcular número de filas
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    axs = axs.flatten()

    for i, (attr, title) in enumerate(attributions):
        axs[i].imshow(original_image)
        axs[i].imshow(attr, cmap=cmap, alpha=0.6)
        axs[i].axis('off')
        axs[i].set_title(title, fontsize=10)

    # Ocultar los ejes sobrantes
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()


# Cargar el modelo y pesos
checkpoint_files = '/home/anacotler/Documents/vision_artificial/tpf_vision/checkpoints/epoch-14-loss_valid-13.5.pt'
model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name=config['model_name'], pretrain_weights='IMAGENET1K_V2').to(config['device'])
model.load_state_dict(torch.load(checkpoint_files, map_location=config['device']))
model.eval()

# Cargar la imagen
image_path = '/home/anacotler/Documents/vision_artificial/tpf_vision/utkcropped/15_1_0_20170109203912357.jpg.chip.jpg'
image = Image.open(image_path).convert('RGB')

# Preprocesar la imagen
transform = transforms.Compose([
    transforms.Resize((config['img_width'], config['img_height'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['mean'], std=config['std'])
])
input_data = transform(image).unsqueeze(0).to(config['device'])

# Convertir la imagen original a NumPy (denormalizar si es necesario)
original_image = np.array(image) / 255.0  # Normalizar a rango [0, 1]

# Lista para almacenar las atribuciones y títulos
attributions = []

# Iterar sobre las capas convolucionales de ResNet50
for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
    layer = getattr(model.model, layer_name)  # Obtener la capa por su nombre
    for sub_layer_idx, sub_layer in enumerate(layer):
        if isinstance(sub_layer, torch.nn.Module):
            for block_idx, block in enumerate(sub_layer.children()):
                if isinstance(block, torch.nn.Conv2d):  # Verificar si es una capa convolucional
                    # Crear objeto LayerGradCam
                    layer_gc = LayerGradCam(model, block)

                    # Calcular las atribuciones
                    attr = layer_gc.attribute(input_data, target=0)

                    # Preparar las atribuciones para visualización
                    attr_upsampled = LayerAttribution.interpolate(attr, (config['img_width'], config['img_height'])).squeeze(0).detach().cpu().numpy()

                    # Asegurarse de que las dimensiones sean correctas
                    if len(attr_upsampled.shape) == 3 and attr_upsampled.shape[0] == 1:
                        attr_upsampled = attr_upsampled.squeeze(0)

                    # Guardar atribución y título
                    title = f'{layer_name} - Block {sub_layer_idx}-{block_idx}'
                    attributions.append((attr_upsampled, title))

# Mostrar todas las atribuciones en una grilla
plot_attributions_grid(attributions, original_image, cols=4, cmap='jet')
