from config import config
import torch
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from model import AgeEstimationModel

import torchvision.transforms as transforms

    
def inference(model, image_path, output_path):
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
        """
        # Create a PIL image with the age estimation value as text
        output_image = image.copy()
        draw = ImageDraw.Draw(output_image)
        text = f"Age: {age_estimation:.2f}"  
        draw.text((10, 10), text, fill=(255, 0, 0))
        print(f"Age estimation: {age_estimation:.2f}")
        
        # Save the output image
        output_image.save(output_path)
        """
        return age_estimation
"""
path = "/home/anacotler/Documents/vision_artificial/tpf_vision/checkpoints"
checkpoint_files = glob.glob(os.path.join(path, 'epoch-*-loss_valid-*.pt'))
latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name=config['model_name'], pretrain_weights='IMAGENET1K_V2').to(config['device'])
# Load the model using the latest checkpoint file
model.load_state_dict(torch.load(latest_checkpoint, weights_only=True))

image_path_test = config['image_path_test'] # Path to the input image
output_path_test = config['output_path_test']  # Path to save the output image
inference(model, image_path_test, output_path_test)
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# # Dados los pesos de mi modelo, lo cargo, y hago la prediccion de todo el test. Quiero medir el error que obtengo en el test.
path = "/home/anacotler/Documents/vision_artificial/tpf_vision/checkpoints"
#checkpoint_files = glob.glob(os.path.join(path, 'epoch-*-loss_valid-*.pt'))
checkpoint_files = '/home/anacotler/Documents/vision_artificial/tpf_vision/checkpoints/epoch-14-loss_valid-5.74.pt'
#latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name=config['model_name'], pretrain_weights='IMAGENET1K_V2').to(config['device'])
# Load the model using the latest checkpoint file
#model.load_state_dict(torch.load(latest_checkpoint, weights_only=True))
model.load_state_dict(torch.load(checkpoint_files, weights_only=True))

path_csv_dataset = '/home/anacotler/Documents/vision_artificial/tpf_vision/csv_dataset/test_set.csv'
df = pd.read_csv(path_csv_dataset)
df['age'] = df['age'].astype(np.float32)

# hacer la prediccion de todo el test
output_path = ''

reales = []
predicciones = []

outliers_real = []
outliers_pred = []


for i in range(len(df)):
    #print(i)
    image_path = '/home/anacotler/Documents/vision_artificial/tpf_vision/utkcropped/'+df['image_name'][i]
    age = df['age'][i]
    prediction = inference(model, image_path, output_path)

    # Ajustar la prediccion a numeros enteros

    if prediction < 1 and prediction > 0:
        prediction = 1
    elif prediction < 0:
        prediction = 0
    elif prediction > 100:
        prediction = 100
    else:
        prediction = round(prediction)
    
    if np.abs(age - prediction) > 10:
        #print(f"Real: {age}, Predicción: {prediction}")
        #print(f"Image: {image_path}")
        #print('')
        outliers_real.append(age)
        outliers_pred.append(prediction)


    reales.append(age)
    predicciones.append(prediction)

reales = np.array(reales)
predicciones = np.array(predicciones)

outliers_real = np.array(outliers_real)
outliers_pred = np.array(outliers_pred)

plt.scatter(outliers_real, outliers_pred)
plt.xlabel('Real')
plt.ylabel('Predicción')
plt.title(f'Real vs Predicción (Outliers) with {len(outliers_real)} outliers')
plt.show()

# Calcular el error
mae = mean_absolute_error(reales, predicciones)
mse = mean_squared_error(reales, predicciones)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Graficar
plt.scatter(reales, predicciones)
plt.xlabel('Real')
plt.ylabel('Predicción')
plt.title(f'Real vs Predicción (MAE: {mae:.2f})')
plt.show()









