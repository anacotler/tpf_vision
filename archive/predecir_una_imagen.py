from config import config
import torch
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from model import AgeEstimationModel
from model2 import AgeEstimationModel2
from model3 import ResNet50WithAttention

import torchvision.transforms as transforms

    
def inference(model, image_path):
    model.eval() 
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        transform = T.Compose([T.Resize(((config['img_width'], config['img_height']))),
                               T.ToTensor(),
                               T.Normalize(mean=config['mean'], 
                                           std=config['std'])
                              ])
        input_data = transform(image).unsqueeze(0).to(config['device']) 
        outputs = model(input_data)  # Forward pass through the model
        
        # Extract the age estimation value from the output tensor
        age_estimation = outputs.item()

        return age_estimation



# # Dados los pesos de mi modelo, lo cargo, y hago la prediccion de todo el test. Quiero medir el error que obtengo en el test.
path = "/home/anacotler/Documents/vision_artificial/tpf_vision/checkpoints"

checkpoint_files = '/home/anacotler/Documents/vision_artificial/tpf_vision/checkpoints/capa_epoch-45-loss_valid-4.97.pt'


#model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name=config['model_name'], pretrain_weights='IMAGENET1K_V2').to(config['device'])
model = AgeEstimationModel2(input_dim=3, output_nodes=1, model_name='resnet', pretrain_weights='IMAGENET1K_V2').to(config['device'])
#model = ResNet50WithAttention(output_nodes=1).to(config['device'])

model.load_state_dict(torch.load(checkpoint_files, weights_only=True))


image_path = '/home/anacotler/Documents/vision_artificial/tpf_vision/cropped_faces/face_1.jpg'
prediction = inference(model, image_path)

    # Ajustar la prediccion a numeros enteros

if prediction < 1 and prediction > 0:
    prediction = 1
elif prediction < 0:
    prediction = 0
elif prediction > 100:
    prediction = 100
else:
    prediction = round(prediction)

print(f"La prediccion de la edad es: {prediction}")
    
 












