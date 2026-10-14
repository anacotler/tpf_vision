import torch
import torch.nn as nn
from torchvision.models import resnet50

class SelfAttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionLayer, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Generar Query, Key y Value
        query = self.query(x).view(batch_size, -1, height * width)  # [B, C/8, H*W]
        key = self.key(x).view(batch_size, -1, height * width)  # [B, C/8, H*W]
        value = self.value(x).view(batch_size, -1, height * width)  # [B, C, H*W]

        # Atención: pesos = softmax(query^T * key)
        attention = self.softmax(torch.bmm(query.permute(0, 2, 1), key))  # [B, H*W, H*W]

        # Aplicar pesos de atención a Value
        out = torch.bmm(value, attention)  # [B, C, H*W]
        out = out.view(batch_size, channels, height, width)  # [B, C, H, W]

        return out



class ResNet50WithAttention(nn.Module):
    def __init__(self, output_nodes, pretrain_weights='IMAGENET1K_V2'):
        super(ResNet50WithAttention, self).__init__()
        # Cargar la arquitectura base de ResNet-50
        self.model = resnet50(weights=pretrain_weights)
        
        # Inserción de Self-Attention después de conv1
        self.self_attention = SelfAttentionLayer(in_channels=64)
        
        # Modificar la capa fully connected (fc)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=2048, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=output_nodes, bias=True)
        )
    
    def forward(self, x):
        # Paso inicial: conv1, BN, ReLU, MaxPool
        x = self.model.conv1(x)  # [B, 64, H, W]
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # Aplicar atención en las características de la primera capa
        x = self.self_attention(x)  # [B, 64, H, W]
        
        # Pasar por los bloques residuales de ResNet
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # Global Average Pooling (GAP)
        x = self.model.avgpool(x)  # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)  # [B, 2048]

        # Clasificación final
        x = self.model.fc(x)
        return x

