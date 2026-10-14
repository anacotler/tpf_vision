"""Entrena AgeResNet (ResNet-101 + cabeza DLDL) sobre UTKFace.

Por defecto usa el split de data/processed/ (train_set / valid_set / test_set),
que es con el que se entreno arista: los train.csv/validation.csv/test.csv de la
raiz son OTRO split y su test incluye ~2400 caras que arista vio al entrenar. Guarda el mejor modelo
segun el MAE de validacion en models/resnet101.pt y la historia por epoca
en reports/training/historia_resnet101.csv. Se corre desde la raiz del repo:

    python -m age_estimation.training.train_resnet101                          # 40 epocas, 200 px
    python -m age_estimation.training.train_resnet101 --aug fuerte --epocas 30 \
        --nombre resnet101_robusta --salida models/resnet101_robusta.pt  # la de la demo
    python -m age_estimation.training.train_resnet101 --epocas 2 --limite 512   # prueba rapida
"""
import argparse
import io
import math
import random
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from age_estimation.models.resnet101 import EDAD_MAX, AgeResNet

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class UTKFace(Dataset):
    def __init__(self, csv, raiz, transform, limite=None):
        self.df = pd.read_csv(csv)
        if limite:
            self.df = self.df.iloc[:limite]
        self.raiz = raiz
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        fila = self.df.iloc[i]
        img = Image.open(os.path.join(self.raiz, fila['image_name'])).convert('RGB')
        return self.transform(img), torch.tensor(float(fila['age']))


class JPEGAleatorio:
    """Recomprime en JPEG con calidad al azar, como hace la webcam."""

    def __init__(self, calidad=(30, 90), p=0.5):
        self.calidad, self.p = calidad, p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=random.randint(*self.calidad))
        buf.seek(0)
        return Image.open(buf).convert('RGB')


def transforms(tam, aug='normal'):
    if aug == 'fuerte':
        # Imita lo que cambia entre frames de webcam: el encuadre (jitter de los
        # landmarks de YuNet), la luz, el foco y la compresion. Son justo las
        # perturbaciones que en el diagnostico movian la edad hasta 15+ anios.
        train = T.Compose([
            T.RandomResizedCrop(tam, scale=(0.75, 1.0), ratio=(0.88, 1.12)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.RandomAffine(degrees=10, translate=(0.04, 0.04))], p=0.7),
            T.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3, hue=0.04),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
            JPEGAleatorio(),
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
        evaluacion = T.Compose([T.Resize((tam, tam)), T.ToTensor(), T.Normalize(MEAN, STD)])
        return train, evaluacion
    train = T.Compose([
        T.RandomResizedCrop(tam, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomRotation(10)], p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    evaluacion = T.Compose([T.Resize((tam, tam)), T.ToTensor(), T.Normalize(MEAN, STD)])
    return train, evaluacion


def etiquetas_blandas(edades, sigma):
    """Gaussiana normalizada sobre 0..EDAD_MAX centrada en cada edad real."""
    soporte = torch.arange(EDAD_MAX + 1, device=edades.device, dtype=torch.float32)
    p = torch.exp(-0.5 * ((soporte[None] - edades[:, None]) / sigma) ** 2)
    return p / p.sum(dim=1, keepdim=True)


def perdida(modelo, logits, edades, sigma, peso_l1):
    # DLDL-v2: KL contra la distribucion blanda + L1 sobre la edad esperada.
    log_p = F.log_softmax(logits.float(), dim=1)
    kl = F.kl_div(log_p, etiquetas_blandas(edades, sigma), reduction='batchmean')
    l1 = (modelo.edad_esperada(logits) - edades[:, None]).abs().mean()
    return kl + peso_l1 * l1


@torch.no_grad()
def evaluar(modelo, loader):
    """MAE con TTA de espejado (lo mismo que hace la demo)."""
    modelo.eval()
    err, n = 0.0, 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(DEVICE)
        with torch.autocast('cuda', dtype=torch.float16, enabled=DEVICE == 'cuda'):
            pred = (modelo(x) + modelo(torch.flip(x, dims=[3]))) / 2
        err += (pred.float().squeeze(1) - y).abs().sum().item()
        n += len(y)
    return err / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='resnet101')
    ap.add_argument('--raiz', default='data/raw/UTKFace')
    ap.add_argument('--train-csv', default='data/processed/train_set.csv')
    ap.add_argument('--valid-csv', default='data/processed/valid_set.csv')
    ap.add_argument('--tam', type=int, default=200)
    ap.add_argument('--epocas', type=int, default=40)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4, help="lr de la cabeza; el backbone usa lr/10")
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--sigma', type=float, default=2.0)
    ap.add_argument('--peso-l1', type=float, default=1.0)
    ap.add_argument('--salida', default='models/resnet101.pt')
    ap.add_argument('--limite', type=int, default=None, help="usar solo N imagenes (debug)")
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--aug', default='normal', choices=['normal', 'fuerte'],
                    help="fuerte: agrega jitter de encuadre, blur y JPEG (robustez en webcam)")
    ap.add_argument('--nombre', default=None,
                    help="sufijo para reports/training/historia_<nombre>.csv (por defecto, la arch)")
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    t_train, t_eval = transforms(args.tam, args.aug)
    train = UTKFace(args.train_csv, args.raiz, t_train, args.limite)
    valid = UTKFace(args.valid_csv, args.raiz, t_eval, args.limite)
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, drop_last=True, **kw)
    valid_loader = DataLoader(valid, batch_size=128, **kw)

    modelo = AgeResNet(args.arch).to(DEVICE).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW([
        {'params': modelo.backbone.parameters(), 'lr': args.lr / 10},
        {'params': modelo.fc.parameters(), 'lr': args.lr},
    ], weight_decay=args.wd)

    # Warmup lineal de 1 epoca y despues coseno hasta 0.
    pasos_totales = args.epocas * len(train_loader)
    pasos_warmup = len(train_loader)
    def factor(paso):
        if paso < pasos_warmup:
            return (paso + 1) / pasos_warmup
        avance = (paso - pasos_warmup) / max(1, pasos_totales - pasos_warmup)
        return 0.5 * (1 + math.cos(math.pi * avance))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    scaler = torch.amp.GradScaler(enabled=DEVICE == 'cuda')

    os.makedirs(os.path.dirname(args.salida) or '.', exist_ok=True)
    os.makedirs('reports/training', exist_ok=True)
    historia, mejor = [], float('inf')
    for epoca in range(args.epocas):
        modelo.train()
        t0, suma, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(DEVICE, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=DEVICE == 'cuda'):
                logits = modelo.logits(x)
            loss = perdida(modelo, logits, y, args.sigma, args.peso_l1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            suma += loss.item() * len(y)
            n += len(y)

        mae = evaluar(modelo, valid_loader)
        guardado = mae < mejor
        if guardado:
            mejor = mae
            torch.save(modelo.state_dict(), args.salida)
        historia.append({'epoca': epoca + 1, 'loss_train': suma / n, 'mae_valid': mae})
        pd.DataFrame(historia).to_csv(f'reports/training/historia_{args.nombre or args.arch}.csv',
                                      index=False)
        print(f"epoca {epoca + 1:2d}/{args.epocas}  loss {suma / n:.3f}  "
              f"MAE valid {mae:.3f}{'  *' if guardado else ''}  ({time.time() - t0:.0f}s)",
              flush=True)

    print(f"[ok] Mejor MAE de validacion: {mejor:.3f}  ->  {args.salida}")


if __name__ == '__main__':
    main()
