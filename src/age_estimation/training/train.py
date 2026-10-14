import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchmetrics as tm

import os
import matplotlib.pyplot as plt

from age_estimation.config import config
from age_estimation.models.frozen import AgeEstimationModel
from age_estimation.models.arista import AgeEstimationModel as AgeEstimationModel2
from age_estimation.models.self_attention import ResNet50WithAttention
from age_estimation.training.engine import train_one_epoch, validation
from age_estimation.dataset import train_loader, valid_loader


# create a folder to save checkpoints
path = 'models'
try:
    os.mkdir(path)
except OSError as error:
    print(error)

# Step 5: Train model for longer epochs using the best model from step 4 in hyperparameters_tuning.
# Define model, define optimizer and Set learning rate and weight decay.
#model = AgeEstimationModel2(input_dim=3, output_nodes=1, model_name='resnet', pretrain_weights='IMAGENET1K_V2').to(config['device'])
model = ResNet50WithAttention(output_nodes=1).to(config['device'])
loss_fn = nn.L1Loss()
metric = tm.MeanAbsoluteError().to(config['device'])    
#optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['wd'])
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], momentum=0.9, weight_decay=config['wd'])


# Define checkpoint path (si existe)
#checkpoint_path = '/home/anacotler/Documents/vision_artificial/tpf_vision/checkpoints/19EPOCHSALL_epoch-0-loss_valid-5.66.pt' 
#checkpoint_path = '/home/anacotler/Documents/vision_artificial/tpf_vision/checkpoints/capa_epoch-5-loss_valid-8.49.pt'
#checkpoint_path = '/home/anacotler/Documents/vision_artificial/tpf_vision/checkpoints/capa_epoch-34-loss_valid-5.18.pt'
checkpoint_path = ''
# Si existe, carga el modelo y el optimizador
if os.path.exists(checkpoint_path):
    print(f"Cargando checkpoint desde {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
else:
    print("No se encontró un checkpoint. Entrenando desde cero.")


# Write code to train the model for num_epochs epochs.
best_loss = torch.inf
before_model_path = None

loss_train_hist = []
loss_valid_hist = []

metric_train_hist = []
metric_valid_hist = []

writer = SummaryWriter()


for epoch in range(config['epochs']):
    # Train
    model, loss_train, metric_train = train_one_epoch(model,
                                           train_loader,
                                           loss_fn,
                                           optimizer,
                                           metric,
                                           epoch)
    # Validation
    loss_valid, metric_valid = validation(model,
                                          valid_loader,
                                          loss_fn, 
                                          metric)

    loss_train_hist.append(loss_train)
    loss_valid_hist.append(loss_valid)

    metric_train_hist.append(metric_train)
    metric_valid_hist.append(metric_valid)

    if loss_valid < best_loss:
        best_loss = loss_valid
        if before_model_path is not None:
            os.remove(before_model_path)
        before_model_path = f'models/capa_cong_epoch-{epoch}-loss_valid-{best_loss:.3}.pt'
        torch.save(model.state_dict(), before_model_path)
        print(f'\nModel saved in epoch: {epoch}')

    writer.add_scalar('Loss/train', loss_train, epoch)
    writer.add_scalar('Loss/test', loss_valid, epoch)
    

    print()
    print(f'Train: loss={loss_train:.3}')
    print(f'Valid: loss={loss_valid:.3}')
    print()

    #if epoch % 5 == 0:
    #    print()
    #    print(f'Train: loss={loss_train:.3}')
    #    print(f'Valid: loss={loss_valid:.3}')
    #    print()


writer.close()

# Plots: Plot learning curves
plt.plot(range(config['epochs']), loss_train_hist, 'r-', label='Train-loss')
plt.plot(range(config['epochs']), loss_valid_hist, 'b-', label='Validation-loss')
plt.plot(range(config['epochs']), metric_train_hist, 'g-', label='Train-metric')
plt.plot(range(config['epochs']), metric_valid_hist, 'b-', label='Validation-metric')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend()
plt.savefig('./figs/metric_plot.png')