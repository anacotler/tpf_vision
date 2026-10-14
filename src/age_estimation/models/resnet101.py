import torch
import torch.nn as nn
import torchvision.models as tvm


EDAD_MAX = 100   # la cabeza predice una distribucion sobre 0..EDAD_MAX


class AgeResNet(nn.Module):
    """ResNet (101 por defecto) con cabeza de distribucion de edad (DLDL).

    En vez de regresar un numero, la red da logits sobre las edades 0..100 y la
    edad es el valor esperado de esa distribucion. Entrenar contra etiquetas
    "blandas" (una gaussiana centrada en la edad real) tolera mejor el ruido de
    etiquetas de UTKFace que la regresion L1 pura.

    `forward` devuelve la edad esperada con forma [B, 1], igual que los modelos
    viejos, asi la demo y la explicabilidad lo usan sin cambios. El backbone es un
    nn.Sequential con el mismo orden que models/arista.py (layer1..4 en 4..7), que es
    lo que espera `explicabilidad.resolver_capa`.
    """

    def __init__(self, arch='resnet101', pretrain_weights='IMAGENET1K_V2'):
        super().__init__()
        base = getattr(tvm, arch)(weights=pretrain_weights)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(base.fc.in_features, EDAD_MAX + 1),
        )
        self.register_buffer('edades', torch.arange(EDAD_MAX + 1, dtype=torch.float32))

    def logits(self, x):
        return self.fc(torch.flatten(self.backbone(x), 1))

    def edad_esperada(self, logits):
        return (logits.float().softmax(dim=1) * self.edades).sum(dim=1, keepdim=True)

    def forward(self, x):
        return self.edad_esperada(self.logits(x))
