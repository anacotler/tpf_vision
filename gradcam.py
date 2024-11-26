import torch
import torchvision.transforms as transforms
from captum.attr import LayerGradCam, visualize_image_attr
from captum.attr import LayerAttribution
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import AgeEstimationModel
from config import config
import os
from torchvision import models

# Visualize attribution scores with Matplotlib
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


# # Dados los pesos de mi modelo, lo cargo, y hago la prediccion de todo el test. Quiero medir el error que obtengo en el test.
path = "/home/anacotler/Documents/vision_artificial/tpf_vision/checkpoints"
checkpoint_files = '/home/anacotler/Documents/vision_artificial/tpf_vision/checkpoints/epoch-14-loss_valid-13.5.pt'
model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name=config['model_name'], pretrain_weights='IMAGENET1K_V2').to(config['device'])
model.load_state_dict(torch.load(checkpoint_files, weights_only=True))

# # Load the model
# model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name=config['model_name'], pretrain_weights='IMAGENET1K_V2').to(config['device'])
# model.load_state_dict(torch.load('checkpoints/epoch-9-loss_valid-0.09.pt', map_location=config['device']))
# model.eval()

# Load the image
image_path = 'img_test/1_0_0_20161219140623097.jpg'
image = Image.open(image_path).convert('RGB')

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((config['img_width'], config['img_height'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['mean'], std=config['std'])
])
input_data = transform(image).unsqueeze(0).to(config['device'])

# Target layer for Grad-CAM (last convolutional layer of ResNet50)
target_layer = model.model.layer4[-1]

# Create a LayerGradCam object
layer_gc = LayerGradCam(model.model, target_layer)

# Compute the attribution scores
attr = layer_gc.attribute(input_data, target=0)

# Prepare attribution for visualization
attr_upsampled = LayerAttribution.interpolate(attr, (config['img_width'], config['img_height'])).squeeze(0).cpu().numpy()

# Convert original image to NumPy (denormalize if needed)
original_image = np.array(image) / 255.0  # Normalize to [0, 1] range

# Visualize the attribution scores
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



