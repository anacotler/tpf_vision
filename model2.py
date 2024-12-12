import torch
import torch.nn as nn
from torchvision.models import resnet50


class AgeEstimationModel2(nn.Module):
    def __init__(self, input_dim, output_nodes, model_name, pretrain_weights):
        super(AgeEstimationModel2, self).__init__()

        self.input_dim = input_dim
        self.output_nodes = output_nodes
        self.pretrain_weights = pretrain_weights

        if model_name == 'resnet':
            # Cargar modelo base
            base_model = resnet50(weights=pretrain_weights)

            #for param in base_model.parameters():
            #    param.requires_grad = False
            
            # Hook para capturar la salida de la primera capa (conv1)
            self.first_layer_output = None
            base_model.conv1.register_forward_hook(self.save_first_layer_output)
            
            # Dividir el modelo en cuerpo (sin la fc) y la fc modificada
            self.backbone = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
                base_model.avgpool,  # Salida de características finales (2048)
            )
            
            # Modificar la capa fully connected para incluir características adicionales
            custom_feature_size = 64  # Número de filtros en conv1 (ResNet50)
            self.fc = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=2048 + custom_feature_size, out_features=256, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=self.output_nodes, bias=True)
            )

    def save_first_layer_output(self, module, input, output):
        """Guardar la salida de la primera capa (conv1)."""
        self.first_layer_output = output

    def forward(self, x):
        # Forward a través del cuerpo de ResNet
        features = self.backbone(x)  # Salida antes de la capa fc (2048)
        features = torch.flatten(features, 1)  # Aplanar para concatenación
        
        # Procesar las características de la primera capa (conv1)
        if self.first_layer_output is not None:
            first_layer_features = self.first_layer_output.mean(dim=(2, 3))  # GAP -> [batch_size, 64]
        else:
            raise ValueError("La salida de la primera capa no fue capturada. Verifica el hook.")
        
        # Concatenar las características
        combined_features = torch.cat([features, first_layer_features], dim=1)  # [batch_size, 2048 + 64]
        
        # Pasar a través de la capa fully connected
        out = self.fc(combined_features)
        return out
