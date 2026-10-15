"""
Demo en vivo de estimacion de edad — Semana de la Salud Mental (UdeSA).

Toma la webcam, detecta caras, muestra la edad estimada en tiempo real y, en un
panel al costado, *que parte de la cara* esta mirando la red para decidir ese
numero (Grad-CAM u oclusion, ver explicabilidad.py).

El recorte por defecto es ALINEADO: usa los 5 landmarks de YuNet para reproducir
el encuadre de UTKFace (ver PLANTILLA_UTK). Importa mucho mas de lo que parece
—el modelo se entreno con caras alineadas y es muy sensible al encuadre—; sobre
data/samples/ el error medio baja de 9.6 a 1.6 anios, y una cara de 73 pasa de
predecirse 33 a 71. Con --recorte cuadrado se vuelve al recorte con margen.

Uso tipico (con la webcam):
    python -m age_estimation.demo

Probar sobre una imagen (sin webcam), guarda la ventana entera a disco:
    python -m age_estimation.demo --image data/samples/40_0_2.jpg

Por defecto usa --modelo combinado: promedia arista (ResNet-50, 128 px) y la
ResNet-101 a 200 px entrenada con aumentacion tipo webcam
(training/train_resnet101.py --aug fuerte). En el test de data/processed/ baja
el MAE de 5.11 a 4.78 y es el mas estable frente a ruido tipo webcam. Para un solo modelo:
    python -m age_estimation.demo --modelo arista
    python -m age_estimation.demo --modelo resnet101

Estabilidad en vivo (con el suavizado prendido, tecla 's'): los landmarks se
filtran en el tiempo antes de recortar, la edad se promedia sobre hasta 5 recortes
levemente corridos + sus espejos (--recortes-tta, 1 por defecto), y despues se suaviza con EMA o mediana
(--suavizado mediana --ventana 9).

Teclas (tambien estan en pantalla con 'h'):
    q / ESC  salir                     e  explicacion on/off
    espacio  congelar la imagen        m  metodo: Grad-CAM <-> oclusion
    s        suavizado temporal        l  capa del Grad-CAM: layer3 <-> layer4
             (edad + mapa + barras)
    f        pantalla completa         o  pintar el mapa también sobre el video
    g        guardar captura           r  contornos de las regiones
"""
import argparse
import datetime as dt
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from age_estimation.demo import ui
from age_estimation.explainability import (GradCAM, atribucion_por_region, dibujar_contornos_regiones,
                            landmarks_estimados, mapa_oclusion, mascaras_regiones,
                            normalizar, resolver_capa, superponer, ventana_ovalo)
from age_estimation.models.arista import AgeEstimationModel as AristaModel

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
IMG_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
    else 'cpu'
)

# Raiz del repo: las rutas por defecto (models/, reports/captures/) salen de aca y no
# del directorio desde donde se lance, asi la demo anda desde cualquier lado.
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Modelo YuNet (opcional pero muy recomendado: ademas de detectar mejor, da los
# 5 landmarks que usamos para nombrar las regiones de la cara).
YUNET_PATH = os.path.join(RAIZ, 'models', 'face_detection_yunet_2023mar.onnx')

# Escala de toda la interfaz: letra, panel, margenes, espaciados y grosores.
# Todas las medidas de dibujo estan escritas para ESCALA_UI = 1 y pasan por
# esc(), asi que cambiando este numero (--escala-ui) crece todo junto y nada se
# superpone. Lo fija `aplicar_escala_ui`, que recalcula tambien las constantes
# de abajo.
ESCALA_UI = 1.0


def esc(valor):
    """Medida de la interfaz escrita para ESCALA_UI = 1 -> pixeles reales."""
    return max(1, int(round(valor * ESCALA_UI)))


ANCHO_PANEL = 400          # ancho del panel de explicacion, a la derecha
ALTO_BARRA = 58            # franja inferior con el disclaimer
ALTO_MIN = 730             # alto minimo del lienzo, para que entre el panel
TAM_EXPL = 260             # lado en el que se calcula y se muestra el mapa


def aplicar_escala_ui(escala):
    global ESCALA_UI, ANCHO_PANEL, ALTO_BARRA, ALTO_MIN, TAM_EXPL
    ESCALA_UI = escala
    ANCHO_PANEL, ALTO_BARRA, ALTO_MIN, TAM_EXPL = esc(400), esc(58), esc(730), esc(260)

CARPETA_CAPTURAS = os.path.join(RAIZ, 'reports', 'captures')

# Plantilla de alineacion de UTKFace, en fracciones del lado del recorte.
# UTKFace no son fotos crudas: son "chips" ya alineados, y por eso los 5 puntos
# caen casi siempre en el mismo lugar. Estos valores salen de correr YuNet sobre
# los chips de data/samples/ (desvio de 0.012-0.019 del lado en los ojos).
# Reproducir este encuadre en la webcam es lo que hace que el modelo vea lo
# mismo que vio entrenando; con el recorte cuadrado a ojo el error se triplica.
PLANTILLA_UTK = np.array([
    [0.300, 0.298],   # ojo derecho
    [0.691, 0.298],   # ojo izquierdo
    [0.491, 0.529],   # nariz
    [0.323, 0.695],   # comisura derecha
    [0.665, 0.695],   # comisura izquierda
], dtype=np.float32)
TAM_ALINEADO = 200    # lado nativo de los chips de UTKFace

_transforms = {}

# Bajarle la nitidez al recorte (200 px) antes de pasarlo a la red.
# Las caras de UTKFace son blandas (muchas son fotos chicas reescaladas): su
# nitidez (varianza del Laplaciano) tiene mediana ~36, mientras que una webcam
# a 1 m da 300-700. Ante ese detalle extra la red lee "piel lisa de chico":
# a una persona de 27 la ResNet-101 le daba ~14.
#
# Por defecto se ACHICA el recorte a 64 px (y la red lo vuelve a agrandar).
# Sobre 6 capturas de la misma persona en distintas poses dio 18-26 anios
# (con blur sigma=1.0 daba 16-32), a costa de subir el MAE en UTKFace de
# 4.95 a 5.15. El blur queda como alternativa (--blur 1.0 --reducir 0).
REDUCIR_ENTRADA = 64
SIGMA_ENTRADA = 0.0

# Las fotos de UTKFace tienen mas contraste que una webcam con luz pareja de
# oficina, y con la imagen "chata" la red tambien tira la edad para abajo.
# Estirar la luminancia de la cara (percentiles 2-98) subio la edad promedio
# de 20.1 a 22.7 en 12 capturas de una persona de 27 y achico la diferencia
# entre poses; en UTKFace el MAE pasa de 5.24 a 5.37.
AUTOCONTRASTE = True


class AutoContraste:
    """Estira la luminancia (canal L de Lab) al rango de percentiles 2-98 del
    centro del recorte, que es donde esta la cara; el color no se toca."""

    def __call__(self, pil):
        lab = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2LAB)
        lum = lab[..., 0].astype(np.float32)
        alto, ancho = lum.shape
        centro = lum[alto // 5:alto * 4 // 5, ancho // 5:ancho * 4 // 5]
        bajo, alto_p = np.percentile(centro, (2, 98))
        lab[..., 0] = np.clip((lum - bajo) / max(alto_p - bajo, 1.0) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def transform_de(modelo):
    """Preprocesamiento del modelo: cada uno se entreno a su tamano de entrada
    (arista a 128 px, la ResNet-101 a 200), guardado en `modelo.tam_entrada`."""
    tam = getattr(modelo, 'tam_entrada', IMG_SIZE)
    clave = (tam, SIGMA_ENTRADA, REDUCIR_ENTRADA, AUTOCONTRASTE)
    if clave not in _transforms:
        contraste = [AutoContraste()] if AUTOCONTRASTE else []
        blur = ([T.GaussianBlur(kernel_size=2 * math.ceil(3 * SIGMA_ENTRADA) + 1,
                                sigma=SIGMA_ENTRADA)] if SIGMA_ENTRADA > 0 else [])
        reducir = [T.Resize((REDUCIR_ENTRADA, REDUCIR_ENTRADA))] if REDUCIR_ENTRADA > 0 else []
        _transforms[clave] = T.Compose(contraste + blur + reducir + [
            T.Resize((tam, tam)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])
    return _transforms[clave]

NOTAS_METODO = {
    'gradcam': ("Pesa las activaciones de una capa convolucional",
                "por su gradiente. Correlacional, no causal."),
    'oclusion': ("Tapa un parche de la cara y mide cuanto se",
                 "mueve la edad. Causal, pero cuesta 64 pasadas."),
}

AYUDA = [
    ("q / ESC", "salir"),
    ("espacio", "congelar la imagen"),
    ("e", "explicación on / off"),
    ("m", "método: Grad-CAM ↔ oclusión"),
    ("l", "capa del Grad-CAM: layer3 ↔ layer4"),
    ("o", "pintar el mapa también sobre el video"),
    ("r", "contornos de las regiones"),
    ("s", "suavizado temporal: edad, mapa y barras"),
    ("f", "pantalla completa"),
    ("g", "guardar captura en reports/captures/"),
]


# ----------------------------------------------------------------------------
# Deteccion de caras
# ----------------------------------------------------------------------------
@dataclass
class Deteccion:
    caja: tuple                       # (x, y, w, h) en pixeles del frame
    landmarks: Optional[np.ndarray] = None   # (5, 2) en pixeles del frame
    score: float = 1.0

    @property
    def area(self):
        return self.caja[2] * self.caja[3]


class FaceDetector:
    """Detector de caras. Usa YuNet (robusto, y devuelve landmarks) si el
    modelo onnx esta presente; si no, cae a Haar cascade (peor, sin landmarks,
    pero viene con OpenCV)."""

    def __init__(self, prefer='auto', ancho_max=640):
        self.mode = None
        self.yunet = None
        self.haar = None
        # YuNet corre sobre el frame achicado a este ancho. A 1280x720 tarda ~36
        # ms en CPU y a 640 ~5 ms, y los landmarks salen igual de precisos
        # (medido contra la plantilla de UTKFace: 0.07 vs 0.09 de la distancia
        # entre ojos), asi que no tiene sentido pagar la resolucion completa.
        self.ancho_max = ancho_max

        want_yunet = prefer in ('auto', 'yunet') and hasattr(cv2, 'FaceDetectorYN')
        if want_yunet and os.path.exists(YUNET_PATH):
            self.yunet = cv2.FaceDetectorYN.create(
                YUNET_PATH, "", (320, 320), 0.7, 0.3, 5000
            )
            self.mode = 'yunet'
        else:
            if prefer == 'yunet':
                print(f"[!] No encontre {YUNET_PATH}; uso Haar cascade.")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haar = cv2.CascadeClassifier(cascade_path)
            if self.haar.empty():
                print("[x] Tu instalacion de OpenCV no trae los cascades de Haar "
                      "(pasa en OpenCV 5.x).")
                print(f"    Baja YuNet a {YUNET_PATH} o instala 'opencv-python<5'.")
                sys.exit(1)
            self.mode = 'haar'
        print(f"[i] Detector de caras: {self.mode}"
              f"{' (sin landmarks)' if self.mode == 'haar' else ''}")

    @property
    def da_landmarks(self):
        return self.mode == 'yunet'

    def detect(self, frame_bgr):
        """Devuelve lista de Deteccion."""
        if self.mode == 'yunet':
            alto, ancho = frame_bgr.shape[:2]
            escala = min(1.0, self.ancho_max / ancho) if self.ancho_max else 1.0
            if escala < 1.0:
                frame_bgr = cv2.resize(frame_bgr, None, fx=escala, fy=escala,
                                       interpolation=cv2.INTER_AREA)
                alto, ancho = frame_bgr.shape[:2]
            self.yunet.setInputSize((ancho, alto))
            _, caras = self.yunet.detect(frame_bgr)
            if caras is None:
                return []
            salida = []
            for cara in caras:
                # Formato YuNet: x, y, w, h, 5 landmarks (x,y), score.
                # Landmarks: ojo der, ojo izq, nariz, comisura der, comisura izq.
                # Se devuelven en pixeles del frame ORIGINAL.
                caja = tuple(int(round(v / escala)) for v in cara[:4])
                landmarks = np.asarray(cara[4:14], dtype=np.float32).reshape(5, 2) / escala
                salida.append(Deteccion(caja, landmarks, float(cara[14])))
            return salida
        # Haar
        gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        caras = self.haar.detectMultiScale(
            gris, scaleFactor=1.2, minNeighbors=8, minSize=(80, 80)
        )
        return [Deteccion(tuple(int(v) for v in c)) for c in caras]


# ----------------------------------------------------------------------------
# Recorte
# ----------------------------------------------------------------------------
@dataclass
class Recorte:
    """Recorte cuadrado de la cara y la transformacion que lo produjo.

    `M` es la afin 2x3 que lleva del frame al recorte. Guardarla (en vez de solo
    un (x0, y0)) permite que el recorte alineado —que ademas rota y escala— use
    exactamente el mismo camino que el recorte cuadrado para mapear landmarks
    hacia adentro y el mapa de calor de vuelta al video.
    """
    imagen: np.ndarray    # BGR, lado x lado
    M: np.ndarray         # afin 2x3: frame -> recorte
    lado: int
    alineado: bool = False

    def a_coords(self, puntos_frame, tam):
        """Puntos en pixeles del frame -> pixeles de una version de `tam` px."""
        puntos = np.asarray(puntos_frame, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.transform(puntos, self.M).reshape(-1, 2) * (tam / self.lado)


def recortar_cara(frame_bgr, caja, margen=0.2):
    """Recorta la cara con margen y la deja CUADRADA, para que el encuadre se
    parezca al del dataset de entrenamiento (UTKFace).

    Si la caja se pasa del borde del frame se rellena replicando el borde, en
    lugar de recortar contra el limite: recortando, el parche deja de ser
    cuadrado y el Resize((128,128)) posterior estira la cara justo cuando la
    persona se corre del centro, que es cuando peor anda el modelo.
    """
    x, y, ancho, alto = caja
    cx, cy = x + ancho / 2.0, y + alto / 2.0
    lado = int(round(max(ancho, alto) * (1 + 2 * margen)))
    if lado < 8:
        return None
    x0 = int(round(cx - lado / 2.0))
    y0 = int(round(cy - lado / 2.0))

    alto_f, ancho_f = frame_bgr.shape[:2]
    rx0, ry0 = max(0, x0), max(0, y0)
    rx1, ry1 = min(ancho_f, x0 + lado), min(alto_f, y0 + lado)
    if rx1 <= rx0 or ry1 <= ry0:
        return None

    parche = frame_bgr[ry0:ry1, rx0:rx1]
    izq, arriba = rx0 - x0, ry0 - y0
    der, abajo = (x0 + lado) - rx1, (y0 + lado) - ry1
    if izq or der or arriba or abajo:
        parche = cv2.copyMakeBorder(parche, arriba, abajo, izq, der, cv2.BORDER_REPLICATE)
    if parche.shape[0] != lado or parche.shape[1] != lado:
        parche = cv2.resize(parche, (lado, lado))
    M = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0]], dtype=np.float32)
    return Recorte(parche, M, lado)


def recortar_alineado(frame_bgr, landmarks, tam=TAM_ALINEADO):
    """Recorta reproduciendo el encuadre de UTKFace a partir de los landmarks.

    Ajusta una similitud (rotacion + escala uniforme + traslacion, 4 grados de
    libertad) que lleva los 5 puntos detectados a PLANTILLA_UTK. El resultado es
    una cara derecha, del mismo tamano relativo con el que se entreno.

    Se usa LMEDS y no minimos cuadrados porque las comisuras de la boca se
    mueven cuando la persona habla o sonrie; LMEDS las trata como outliers y se
    queda con los puntos estables (ojos y nariz).

    Devuelve None si no hay landmarks o si el ajuste sale absurdo, para que el
    llamador caiga al recorte cuadrado.
    """
    if landmarks is None:
        return None
    origen = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    destino = (PLANTILLA_UTK * tam).astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(origen, destino, method=cv2.LMEDS)
    if M is None or not np.all(np.isfinite(M)):
        return None
    # La escala sale de la parte lineal; si es disparatada, el ajuste fallo.
    escala = float(np.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])))
    if not (1e-3 < escala < 1e3):
        return None
    imagen = cv2.warpAffine(frame_bgr, M, (tam, tam), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    return Recorte(imagen, M.astype(np.float32), tam, alineado=True)


def obtener_recorte(frame_bgr, deteccion, margen=0.2, modo='alineado'):
    """Recorte de la cara segun el modo pedido, con caida a cuadrado.

    'alineado' necesita landmarks, asi que con el detector Haar (que no los da)
    cae solo al recorte cuadrado de siempre.
    """
    if modo == 'alineado':
        recorte = recortar_alineado(frame_bgr, deteccion.landmarks)
        if recorte is not None:
            return recorte
    return recortar_cara(frame_bgr, deteccion.caja, margen)


def a_tensor(recorte_bgr, modelo):
    """Recorte BGR -> tensor [1, 3, tam, tam] normalizado para `modelo`, en el device."""
    rgb = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB)
    return transform_de(modelo)(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)


# ----------------------------------------------------------------------------
# Modelo(s)
# ----------------------------------------------------------------------------
def load_arista(weights_path):
    model = AristaModel(
        input_dim=3, output_nodes=1, model_name='resnet',
        pretrain_weights='IMAGENET1K_V2'
    ).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(memory_format=torch.channels_last)
    model.tam_entrada = 128
    return model


def load_resnet101(weights_path):
    """ResNet-101 con cabeza de distribucion (models/resnet101.py), entrenada a
    200 px con training/train_resnet101.py."""
    from age_estimation.models.resnet101 import AgeResNet
    model = AgeResNet('resnet101', pretrain_weights=None).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(memory_format=torch.channels_last)
    model.tam_entrada = 200
    return model


def load_finetuneada(weights_path):
    # Arquitectura de model_finetuneado.py, definida inline para no importar ese
    # modulo (arrastra `timm`, que no necesitamos aca). Debe coincidir EXACTO con
    # la cabeza con la que se guardaron los pesos: resnet50 -> Dropout, 2048->256,
    # 256->1 (sin ReLU intermedia).
    from torchvision.models import resnet50

    class Finetuneada(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = resnet50(weights=None)
            self.model.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(2048, 256, bias=True),
                torch.nn.Linear(256, 1, bias=True),
            )

        def forward(self, x):
            return self.model(x)

    model = Finetuneada().to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# Variaciones (escala, dx, dy) del recorte alineado para el TTA multi-recorte,
# en fracciones del lado. El modelo es sensible a corrimientos chicos del
# encuadre (un 2-3% de jitter en los landmarks mueve la edad varios anios);
# promediar sobre estas 5 vistas baja esa sensibilidad ~30% con un solo forward
# (con 9 vistas, ~38%, pero al doble de costo por frame).
VARIANTES_TTA = [(1.00, 0.00, 0.00), (1.04, 0.00, 0.00), (0.96, 0.00, 0.00),
                 (1.00, 0.02, 0.00), (1.00, -0.02, 0.00)]


def variantes_recorte(frame_bgr, recorte, variantes=VARIANTES_TTA):
    """Recortes ligeramente corridos/escalados alrededor del alineado.

    Se re-muestrean desde el frame (componiendo con `recorte.M`) y no desde el
    recorte, asi los bordes traen cara real y no pixeles replicados. Para el
    recorte cuadrado devuelve solo el original.
    """
    if not recorte.alineado:
        return [recorte.imagen]
    lado = recorte.lado
    M = np.vstack([recorte.M, [0, 0, 1]])
    salida = []
    for escala, dx, dy in variantes:
        A = cv2.getRotationMatrix2D((lado / 2, lado / 2), 0, escala)
        A[:, 2] += (dx * lado, dy * lado)
        salida.append(cv2.warpAffine(frame_bgr, (np.vstack([A, [0, 0, 1]]) @ M)[:2],
                                     (lado, lado), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE))
    return salida


@torch.no_grad()
def predecir_edad(models, crops_bgr, tta=True):
    """Predice edad promediando modelos, recortes y, si tta, la imagen espejada.

    `crops_bgr` es un recorte o una lista (ver `variantes_recorte`).

    Devuelve (edad, dispersion). La dispersion es el desvio entre las vistas
    (recortes / espejadas / cada modelo del ensemble). NO es un intervalo de
    confianza: solo dice cuanto se contradice el sistema consigo mismo, que en
    el stand alcanza para mostrar cuando la prediccion es fragil.
    """
    if isinstance(crops_bgr, np.ndarray):
        crops_bgr = [crops_bgr]
    pils = []
    for crop in crops_bgr:
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pils.append(pil)
        if tta:
            pils.append(pil.transpose(Image.FLIP_LEFT_RIGHT))

    vistas = []
    for model in models:
        transform = transform_de(model)
        batch = torch.stack([transform(p) for p in pils]).to(DEVICE)
        batch = batch.to(memory_format=torch.channels_last)
        # fp16 en GPU: ~35% menos tiempo y la edad cambia en centesimas de anio.
        with torch.autocast('cuda', dtype=torch.float16, enabled=DEVICE == 'cuda'):
            salida = model(batch).reshape(-1).float().cpu().numpy()
        vistas.extend(salida.tolist())
    return float(np.mean(vistas)), float(np.std(vistas))


def clamp_age(pred):
    if 0 < pred < 1:
        return 1
    if pred < 0:
        return 0
    if pred > 100:
        return 100
    return round(pred)


# ----------------------------------------------------------------------------
# Suavizado temporal
# ----------------------------------------------------------------------------
class SuavizadorLandmarks:
    """Filtra los landmarks en el tiempo ANTES de recortar.

    YuNet mueve los 5 puntos uno o dos pixeles entre frames aunque la persona
    este quieta; como el recorte alineado sale de esos puntos, el encuadre
    tiembla y la edad con el. Media exponencial por cara, con alpha adaptativo:
    si la cara se mueve rapido (respecto de la distancia entre ojos) el alpha
    sube a 1 y el recorte la sigue sin retraso; quieta, filtra fuerte.
    """

    def __init__(self, alpha=0.3, max_dist=100, ttl=5):
        self.alpha = alpha
        self.max_dist = max_dist
        self.ttl = ttl
        self.tracks = []          # [{'lm': (5, 2), 'miss': int}]

    def update(self, detecciones):
        usados = set()
        for det in detecciones:
            if det.landmarks is None:
                continue
            centro = det.landmarks.mean(axis=0)
            mejor, mejor_d = None, self.max_dist
            for i, t in enumerate(self.tracks):
                d = float(np.linalg.norm(t['lm'].mean(axis=0) - centro))
                if i not in usados and d < mejor_d:
                    mejor, mejor_d = i, d
            if mejor is None:
                self.tracks.append({'lm': det.landmarks.copy(), 'miss': 0})
                usados.add(len(self.tracks) - 1)
                continue
            t = self.tracks[mejor]
            ojos = max(1.0, float(np.linalg.norm(det.landmarks[0] - det.landmarks[1])))
            movimiento = float(np.abs(det.landmarks - t['lm']).max()) / ojos
            alpha = min(1.0, self.alpha + movimiento)
            t['lm'] = alpha * det.landmarks + (1 - alpha) * t['lm']
            t['miss'] = 0
            usados.add(mejor)
            det.landmarks = t['lm'].astype(np.float32)
        for i, t in enumerate(self.tracks):
            if i not in usados:
                t['miss'] += 1
        self.tracks = [t for t in self.tracks if t['miss'] <= self.ttl]


class AgeSmoother:
    """Estabiliza la edad en el tiempo. Asocia cada cara con la del frame
    anterior por cercania del centro y aplica una media exponencial (EMA) o
    la mediana de las ultimas `ventana` predicciones. La mediana ignora los
    saltos aislados (un frame con la cara de costado) en vez de arrastrarlos.
    Asi el numero deja de saltar frame a frame. Se puede apagar para mostrar
    la prediccion cruda (inestable) en el stand."""

    def __init__(self, modo='ema', alpha=0.25, ventana=9, max_dist=100, ttl=12):
        self.modo = modo          # 'ema' o 'mediana'
        self.alpha = alpha        # cuanto pesa el frame nuevo (mas bajo = mas estable)
        self.ventana = ventana    # frames que entran en la mediana
        self.max_dist = max_dist  # distancia max (px) para considerar "la misma cara"
        self.ttl = ttl            # frames que sobrevive una cara sin volver a verse
        self.tracks = {}          # id -> {'c': (cx, cy), 'age': float, 'hist': deque, 'miss': int}
        self._next = 0

    @staticmethod
    def _center(box):
        x, y, w, h = box
        return (x + w / 2, y + h / 2)

    def update(self, detections):
        """detections: [(box, raw_age), ...]  ->  [(box, edad_suavizada), ...]"""
        used = set()
        out = []
        for box, raw in detections:
            cx, cy = self._center(box)
            best, best_d = None, self.max_dist
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                d = ((cx - t['c'][0]) ** 2 + (cy - t['c'][1]) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = tid, d
            if best is None:
                best = self._next
                self._next += 1
                self.tracks[best] = {'c': (cx, cy), 'age': float(raw),
                                     'hist': deque([float(raw)], maxlen=self.ventana),
                                     'miss': 0}
            else:
                t = self.tracks[best]
                if self.modo == 'mediana':
                    t['hist'].append(float(raw))
                    t['age'] = float(np.median(t['hist']))
                else:
                    t['age'] = self.alpha * raw + (1 - self.alpha) * t['age']
                t['c'] = (cx, cy)
                t['miss'] = 0
            used.add(best)
            out.append((box, clamp_age(self.tracks[best]['age'])))
        # Envejecer y limpiar las caras que no se vieron este frame.
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]['miss'] += 1
                if self.tracks[tid]['miss'] > self.ttl:
                    del self.tracks[tid]
        return out


# ----------------------------------------------------------------------------
# Explicacion
# ----------------------------------------------------------------------------
@dataclass
class Explicacion:
    mapa: np.ndarray                     # [TAM_EXPL, TAM_EXPL] en [0, 1]
    miniatura: np.ndarray                # recorte BGR a TAM_EXPL, sin pintar
    ranking: list = field(default_factory=list)   # [(nombre, porcentaje), ...]
    mascaras: dict = field(default_factory=dict)
    metodo: str = 'gradcam'
    capa: str = 'layer4'
    grilla: int = 8
    delta_max: Optional[float] = None    # anios (solo oclusion)
    ms: float = 0.0
    activa: bool = True                  # False = solo la miniatura, sin mapa


class Explicador:
    """Calcula el mapa de "donde mira" para la cara principal, con throttle.

    Cada metodo tiene su intervalo minimo entre recalculos (ver INTERVALOS):
    Grad-CAM es tan barato que corre en cada frame, la oclusion se limita para
    no frenar el video. En el medio se reusa el ultimo mapa.

    La explicacion sale SIEMPRE del primer modelo (arista). Con --ensemble el
    numero promedia los dos modelos, pero el mapa describe solo a arista: un
    mapa promediado entre dos redes distintas no explicaria a ninguna.
    """

    # Medido en MPS (M-series): Grad-CAM ~7 ms, oclusion 8x8 ~62 ms. El Grad-CAM
    # sale gratis y se recalcula siempre; la oclusion se limita para no bajar el
    # video de ~25 fps.
    # Segundos minimos entre mapas. El mapa ya se suaviza en el tiempo, asi que
    # recalcular Grad-CAM a 10 Hz no se nota y libera ~13 ms en la mayoria de
    # los frames.
    INTERVALOS = {'gradcam': 0.1, 'oclusion': 0.15}

    def __init__(self, modelo, capa='layer4', metodo='gradcam', grilla=8,
                 alpha_ranking=0.3, alpha_mapa=0.35):
        self.modelo = modelo
        self.metodo = metodo
        self.capa = capa
        self.grilla = grilla
        self.alpha_ranking = alpha_ranking
        self.alpha_mapa = alpha_mapa
        self.gradcam = GradCAM(modelo, resolver_capa(modelo, capa))
        self.ultima = None
        self._t_ultima = 0.0
        self._ranking_suave = {}
        self._mapa_suave = None
        self._max_suave = 0.0
        self._delta_suave = None
        self._ultimo_centro = None
        self._ultimo_lado = None

    def cambiar_capa(self, capa):
        self.gradcam.cerrar()          # sin esto quedan hooks colgados
        self.capa = capa
        self.gradcam = GradCAM(self.modelo, resolver_capa(self.modelo, capa))
        self.invalidar()

    def alternar_metodo(self):
        self.metodo = 'oclusion' if self.metodo == 'gradcam' else 'gradcam'
        self.invalidar()

    def alternar_capa(self):
        self.cambiar_capa('layer3' if self.capa == 'layer4' else 'layer4')

    def invalidar(self):
        self.ultima = None
        self._t_ultima = 0.0
        self._reiniciar_suavizado()

    def _reiniciar_suavizado(self):
        """Tira el estado de los EMA. Hace falta al cambiar de metodo o de capa
        (los mapas no son comparables entre si) y al cambiar de cara."""
        self._ranking_suave.clear()
        self._mapa_suave = None
        self._max_suave = 0.0
        self._delta_suave = None

    def _misma_cara(self, recorte):
        """Heuristica barata: si el recorte salto de lugar o cambio mucho de
        tamano, es otra persona y no hay que promediar su mapa con el anterior.
        En un stand la gente se turna todo el tiempo delante de la camara."""
        # Centro y tamano del recorte medidos EN EL FRAME. Con el recorte
        # alineado el lado es siempre el mismo (200 px), asi que el tamano hay
        # que sacarlo de la escala de M, no de recorte.lado.
        inversa = cv2.invertAffineTransform(recorte.M)
        medio = recorte.lado / 2.0
        centro = tuple(cv2.transform(
            np.array([[[medio, medio]]], np.float32), inversa).reshape(2))
        escala_m = float(np.sqrt(abs(recorte.M[0, 0] * recorte.M[1, 1]
                                     - recorte.M[0, 1] * recorte.M[1, 0])))
        lado_en_frame = recorte.lado / max(escala_m, 1e-6)

        misma = False
        if self._ultimo_centro is not None and self._ultimo_lado:
            distancia = ((centro[0] - self._ultimo_centro[0]) ** 2
                         + (centro[1] - self._ultimo_centro[1]) ** 2) ** 0.5
            escala = lado_en_frame / self._ultimo_lado
            misma = distancia < 0.5 * lado_en_frame and 0.6 < escala < 1.7
        self._ultimo_centro, self._ultimo_lado = centro, lado_en_frame
        return misma

    def _suavizar_mapa(self, crudo):
        """EMA sobre el mapa crudo Y sobre su maximo.

        No alcanza con suavizar el mapa ya normalizado: como el min-max es por
        frame, un pico espurio en un solo frame reescala todo y eso se ve como
        un parpadeo. Llevando el maximo con su propio EMA, la escala cambia
        despacio y el mapa deja de titilar.
        """
        alpha = self.alpha_mapa
        if self._mapa_suave is None or self._mapa_suave.shape != crudo.shape:
            self._mapa_suave = crudo.copy()
            self._max_suave = float(crudo.max())
        else:
            self._mapa_suave = alpha * crudo + (1 - alpha) * self._mapa_suave
            self._max_suave = alpha * float(crudo.max()) + (1 - alpha) * self._max_suave
        return np.clip(self._mapa_suave / max(self._max_suave, 1e-8), 0.0, 1.0)

    def cerrar(self):
        self.gradcam.cerrar()

    def _suavizar_ranking(self, ranking):
        """EMA sobre los porcentajes: sin esto las barras tiemblan tanto que no
        se pueden leer."""
        for nombre, valor in ranking:
            previo = self._ranking_suave.get(nombre, valor)
            self._ranking_suave[nombre] = self.alpha_ranking * valor + (1 - self.alpha_ranking) * previo
        return sorted(self._ranking_suave.items(), key=lambda par: -par[1])

    def solo_miniatura(self, recorte):
        """Panel con la cara como entra a la red, sin mapa. Se usa cuando la
        explicacion esta apagada, para no pagar el forward/backward de gusto."""
        miniatura = cv2.resize(recorte.imagen, (TAM_EXPL, TAM_EXPL),
                               interpolation=cv2.INTER_AREA)
        return Explicacion(np.zeros((TAM_EXPL, TAM_EXPL), np.float32), miniatura,
                           metodo=self.metodo, capa=self.capa, grilla=self.grilla,
                           activa=False)

    def explicar(self, recorte, landmarks_frame=None, refrescar=True, suavizar=True):
        """Devuelve una Explicacion, reusando la cacheada si todavia sirve.

        `refrescar=False` (imagen congelada) reusa el cache indefinidamente: el
        frame no cambia, asi que recalcular no aporta nada. Un cambio de metodo
        o de capa llama a invalidar() y fuerza el recalculo igual.

        `suavizar=False` (tecla 's') muestra el mapa y el reparto crudos, frame
        a frame. Sirve para mostrar en el stand lo inestable que es la red en
        realidad; con el suavizado prendido se ve mucho mejor pero se pierde esa
        lectura.
        """
        ahora = time.time()
        if self.ultima is not None and (
                not refrescar or ahora - self._t_ultima < self.INTERVALOS[self.metodo]):
            return self.ultima

        if not self._misma_cara(recorte):
            self._reiniciar_suavizado()

        inicio = time.time()
        tensor = a_tensor(recorte.imagen, self.modelo)
        delta_max = None
        if self.metodo == 'gradcam':
            _, mapas = self.gradcam(tensor, normalizar_salida=False)
            crudo = mapas[0]
        else:
            _, crudo, delta_max = mapa_oclusion(self.modelo, tensor, grilla=self.grilla,
                                                normalizar_salida=False)
        crudo = cv2.resize(crudo, (TAM_EXPL, TAM_EXPL), interpolation=cv2.INTER_CUBIC)
        crudo = np.maximum(crudo, 0)   # INTER_CUBIC puede pasarse de rosca y dar negativos

        if suavizar:
            mapa = self._suavizar_mapa(crudo)
            if delta_max is not None:
                previo = self._delta_suave if self._delta_suave is not None else delta_max
                self._delta_suave = self.alpha_mapa * delta_max + (1 - self.alpha_mapa) * previo
                delta_max = self._delta_suave
        else:
            self._reiniciar_suavizado()
            mapa = normalizar(crudo)
        ms = (time.time() - inicio) * 1000.0

        if landmarks_frame is not None:
            landmarks = recorte.a_coords(landmarks_frame, TAM_EXPL)
        else:
            # Sin landmarks (Haar): la cara ocupa el centro del recorte, que es
            # justo lo que el margen del recorte deja definido.
            fraccion = 1.0 / (1 + 2 * 0.2)
            lado_cara = fraccion * TAM_EXPL
            borde = (TAM_EXPL - lado_cara) / 2
            landmarks = landmarks_estimados((borde, borde, lado_cara, lado_cara))

        mascaras = mascaras_regiones(landmarks, TAM_EXPL)
        ranking = atribucion_por_region(mapa, mascaras)
        if suavizar:
            ranking = self._suavizar_ranking(ranking)

        miniatura = cv2.resize(recorte.imagen, (TAM_EXPL, TAM_EXPL),
                               interpolation=cv2.INTER_AREA)
        self.ultima = Explicacion(mapa, miniatura, ranking, mascaras,
                                  self.metodo, self.capa, self.grilla, delta_max, ms)
        self._t_ultima = ahora
        return self.ultima


def superponer_en_frame(frame, recorte, mapa, intensidad=0.85):
    """Pinta el mapa de calor sobre la cara, en el video.

    El mapa vive en coordenadas del recorte, asi que se lleva al frame con la
    inversa de `recorte.M`. Eso cubre tanto el recorte cuadrado (una traslacion)
    como el alineado (que ademas rota y escala), y hace que el calor caiga sobre
    la cara aunque la persona tenga la cabeza inclinada.

    Sobre el video el mapa se limita al ovalo de la cara y lleva un gamma mas
    agresivo que en el panel: aca compite con la escena real y un velo de color
    sobre el fondo se lee como suciedad, no como informacion.
    """
    lado = recorte.lado
    mapa_lado = cv2.resize(mapa, (lado, lado), interpolation=cv2.INTER_CUBIC)
    mapa_lado = mapa_lado * ventana_ovalo(lado)

    inversa = cv2.invertAffineTransform(recorte.M)   # recorte -> frame
    esquinas = np.array([[0, 0], [lado, 0], [lado, lado], [0, lado]],
                        dtype=np.float32).reshape(-1, 1, 2)
    en_frame = cv2.transform(esquinas, inversa).reshape(-1, 2)
    alto_f, ancho_f = frame.shape[:2]
    x0 = max(0, int(np.floor(en_frame[:, 0].min())))
    y0 = max(0, int(np.floor(en_frame[:, 1].min())))
    x1 = min(ancho_f, int(np.ceil(en_frame[:, 0].max())))
    y1 = min(alto_f, int(np.ceil(en_frame[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return

    # Warp directo al ROI: compongo la traslacion en vez de warpear todo el frame.
    al_roi = inversa.copy()
    al_roi[0, 2] -= x0
    al_roi[1, 2] -= y0
    mapa_roi = cv2.warpAffine(mapa_lado, al_roi, (x1 - x0, y1 - y0),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    frame[y0:y1, x0:x1] = superponer(frame[y0:y1, x0:x1], mapa_roi,
                                     intensidad, gamma=2.4)


# ----------------------------------------------------------------------------
# Estado de la interfaz
# ----------------------------------------------------------------------------
@dataclass
class Estado:
    suavizado: bool = True
    explicar: bool = True
    # El mapa de calor va solo en el panel de la derecha. Sobre el video tapa la
    # cara de la persona, que es lo que la gente vino a verse. Se prende con 'o'.
    overlay: bool = False
    contornos: bool = False
    ayuda: bool = False
    congelado: bool = False
    pantalla_completa: bool = False
    fps: Optional[float] = 0.0   # None = no mostrarlo (modo imagen)
    ensemble: bool = False
    recorte: str = 'alineado'    # 'alineado' | 'cuadrado'
    recortes_tta: int = 1        # cuantas VARIANTES_TTA se predicen por frame
    cara_min: int = 120          # ancho minimo (px del frame) para estimar una cara


# ----------------------------------------------------------------------------
# Dibujo
# ----------------------------------------------------------------------------
def etiqueta_edad(edad):
    return f"{edad} año" if edad == 1 else f"{edad} años"


def dibujar_cara(frame, caja, edad, textos, principal=False):
    x, y, ancho, alto = [int(v) for v in caja]
    color = ui.VERDE if principal else ui.TEXTO_TENUE
    ui.caja_esquinas(frame, (x, y, ancho, alto), color, grosor=esc(3 if principal else 2))

    etiqueta = etiqueta_edad(edad)
    tam = esc(30 if principal else 22)
    ancho_txt = ui.ancho_texto(etiqueta, tam, 'bold')
    ancho_pill = ancho_txt + esc(28)
    alto_pill = tam + esc(16)
    # Si no entra arriba de la caja, va adentro.
    py = y - alto_pill - esc(8) if y - alto_pill - esc(8) >= 0 else y + esc(8)
    px = x
    ui.panel_traslucido(frame, (px, py), (px + ancho_pill, py + alto_pill),
                        ui.FONDO_BARRA, alpha=0.80, radio=alto_pill // 2)
    textos.append(ui.Texto(px + ancho_pill // 2, py + alto_pill // 2, etiqueta,
                           tam, color, 'bold', 'mm'))


def dibujar_sin_caras(frame, textos):
    alto, ancho = frame.shape[:2]
    ui.panel_traslucido(frame, (ancho // 2 - esc(220), alto // 2 - esc(44)),
                        (ancho // 2 + esc(220), alto // 2 + esc(44)), ui.FONDO_BARRA,
                        0.70, esc(16))
    textos.append(ui.Texto(ancho // 2, alto // 2 - esc(12), "No veo ninguna cara",
                           esc(26), ui.TEXTO, 'bold', 'mm'))
    textos.append(ui.Texto(ancho // 2, alto // 2 + esc(20), "Acercate y mirá a la cámara",
                           esc(17), ui.TEXTO_TENUE, 'regular', 'mm'))


def dibujar_ayuda(frame, textos):
    alto, ancho = frame.shape[:2]
    alto_caja = esc(62) + len(AYUDA) * esc(30)
    x0, y0 = esc(40), max(esc(20), alto // 2 - alto_caja // 2)
    x1, y1 = x0 + esc(430), y0 + alto_caja
    ui.panel_traslucido(frame, (x0, y0), (x1, y1), ui.FONDO_BARRA, 0.88, esc(16))
    textos.append(ui.Texto(x0 + esc(24), y0 + esc(22), "TECLAS", esc(14), ui.TEXTO_TENUE, 'bold'))
    for i, (tecla, que) in enumerate(AYUDA):
        y = y0 + esc(50) + i * esc(30)
        textos.append(ui.Texto(x0 + esc(24), y, tecla, esc(17), ui.ACENTO, 'bold'))
        textos.append(ui.Texto(x0 + esc(150), y, que, esc(17), ui.TEXTO, 'regular'))


def dibujar_panel(lienzo, x_panel, alto_panel, expl, edad, dispersion, estado, textos):
    """Panel derecho: la cara como la ve la red, el mapa y el reparto por region.

    Las posiciones van con un cursor `y` que baja: el contenido entra en
    ALTO_MIN px, asi que si se agrega una seccion hay que subir esa constante.
    """
    px = x_panel
    lienzo[:alto_panel, px:px + ANCHO_PANEL] = ui.FONDO_PANEL
    izq, der = px + esc(24), px + ANCHO_PANEL - esc(24)

    def linea(y):
        cv2.line(lienzo, (izq, y), (der, y), ui.LINEA, esc(1))

    textos.append(ui.Texto(izq, esc(20), "LO QUE MIRA LA RED", esc(14), ui.TEXTO_TENUE, 'bold'))
    linea(esc(44))

    # --- La cara tal como entra a la red (128x128 reescalado), con el mapa.
    ix, iy = px + (ANCHO_PANEL - TAM_EXPL) // 2, esc(56)
    if expl is not None:
        vista = superponer(expl.miniatura, expl.mapa, 0.85) if expl.activa else expl.miniatura.copy()
        if estado.contornos and expl.mascaras:
            vista = dibujar_contornos_regiones(vista, expl.mascaras, (120, 120, 120), esc(1))
        lienzo[iy:iy + TAM_EXPL, ix:ix + TAM_EXPL] = vista
        cv2.rectangle(lienzo, (ix - 1, iy - 1), (ix + TAM_EXPL, iy + TAM_EXPL), ui.LINEA, esc(1))
    else:
        cv2.rectangle(lienzo, (ix, iy), (ix + TAM_EXPL, iy + TAM_EXPL), (40, 37, 34), -1)
        textos.append(ui.Texto(ix + TAM_EXPL // 2, iy + TAM_EXPL // 2,
                               "esperando una cara", esc(16), ui.TEXTO_TENUE, 'regular', 'mm'))

    # --- Que metodo produjo ese mapa, y cuanto tardo.
    y = iy + TAM_EXPL + esc(18)
    if expl is None:
        textos.append(ui.Texto(izq, y, "Sin cara a la vista", esc(17), ui.TEXTO_TENUE, 'bold'))
        textos.append(ui.Texto(izq, y + esc(24), "La explicación necesita una cara detectada.",
                               esc(13), ui.TEXTO_TENUE))
    elif not expl.activa:
        textos.append(ui.Texto(izq, y, "Explicación apagada", esc(17), ui.TEXTO_TENUE, 'bold'))
        textos.append(ui.Texto(izq, y + esc(24), "Apretá 'e' para ver dónde mira.",
                               esc(13), ui.TEXTO_TENUE))
    else:
        if expl.metodo == 'gradcam':
            titulo, detalle = "Grad-CAM", f" · {expl.capa}"
        else:
            titulo, detalle = "Oclusión", f" · grilla {expl.grilla}×{expl.grilla}"
        textos.append(ui.Texto(izq, y, titulo + detalle, esc(17), ui.ACENTO, 'bold'))
        textos.append(ui.Texto(der, y + esc(2), f"{expl.ms:.0f} ms", esc(14), ui.TEXTO_TENUE,
                               'regular', 'ra'))
        for i, nota in enumerate(NOTAS_METODO[expl.metodo]):
            textos.append(ui.Texto(izq, y + esc(24) + i * esc(17), nota, esc(13), ui.TEXTO_TENUE))

    # --- Reparto del mapa entre las regiones de la cara.
    y += esc(66)
    linea(y)
    y += esc(16)
    textos.append(ui.Texto(izq, y, "DÓNDE MIRA", esc(14), ui.TEXTO_TENUE, 'bold'))
    y += esc(22)
    ranking = expl.ranking if (expl is not None and expl.activa) else []
    if ranking:
        tope = max(valor for _, valor in ranking) or 1.0
        for i, (nombre, valor) in enumerate(ranking[:5]):
            fila = y + i * esc(28)
            destacado = (i == 0)
            color = ui.ACENTO if destacado else ui.TEXTO
            peso = 'medium' if destacado else 'regular'
            textos.append(ui.Texto(izq, fila, nombre, esc(15), color, peso))
            ui.barra(lienzo, px + esc(210), fila + esc(6), esc(118), esc(9), valor / tope,
                     ui.ACENTO if destacado else (110, 105, 100))
            textos.append(ui.Texto(der, fila, f"{valor:.0f}%", esc(15), color, peso, 'ra'))
    else:
        textos.append(ui.Texto(izq, y, "—", esc(15), ui.TEXTO_TENUE))

    # --- El numero, que es lo que la gente vino a ver.
    y += 5 * esc(28) + esc(2)
    linea(y)
    y += esc(16)
    textos.append(ui.Texto(izq, y, "EDAD ESTIMADA", esc(13), ui.TEXTO_TENUE, 'bold'))
    if edad is not None:
        textos.append(ui.Texto(izq, y + esc(18), f"{edad}", esc(46), ui.TEXTO, 'bold'))
        textos.append(ui.Texto(izq + esc(6) + ui.ancho_texto(f"{edad}", esc(46), 'bold'),
                               y + esc(46), "año" if edad == 1 else "años", esc(18),
                               ui.TEXTO_TENUE))
        if dispersion is not None:
            textos.append(ui.Texto(der, y + esc(30), f"±{dispersion:.1f} entre vistas",
                                   esc(13), ui.TEXTO_TENUE, 'regular', 'ra'))
    else:
        textos.append(ui.Texto(izq, y + esc(18), "—", esc(46), ui.TEXTO_TENUE, 'bold'))

    if expl is not None and expl.activa and expl.delta_max is not None:
        textos.append(ui.Texto(
            izq, y + esc(78),
            f"Taparle la zona más sensible le mueve la edad {expl.delta_max:.1f} años.",
            esc(13), ui.TEXTO_TENUE))


def dibujar_barra_inferior(lienzo, y0, estado, textos):
    ancho = lienzo.shape[1]
    lienzo[y0:, :] = ui.FONDO_BARRA
    cv2.line(lienzo, (0, y0), (ancho, y0), ui.LINEA, esc(1))
    textos.append(ui.Texto(
        esc(20), y0 + esc(12),
        "Estimación aproximada. El modelo se entrenó con UTKFace y puede errar bastante "
        "según la luz, el ángulo, la edad y los rasgos de cada cara.",
        esc(15), ui.TEXTO_TENUE))
    estados = [
        f"{estado.fps:.1f} fps" if estado.fps is not None else None,
        DEVICE,
        "ensemble" if estado.ensemble else "1 modelo",
        f"recorte {estado.recorte}",
        f"suavizado {'on' if estado.suavizado else 'off'}",
        "CONGELADO" if estado.congelado else None,
    ]
    textos.append(ui.Texto(esc(20), y0 + esc(34), "  ·  ".join(e for e in estados if e),
                           esc(13), ui.TEXTO_TENUE))
    textos.append(ui.Texto(ancho - esc(20), y0 + esc(34), "h: ayuda  ·  q: salir",
                           esc(13), ui.TEXTO_TENUE, 'regular', 'ra'))


def componer(frame, detecciones, edades, principal, expl, edad_p, dispersion_p, estado):
    """Arma el lienzo completo: video + panel + barra inferior."""
    alto_v, ancho_v = frame.shape[:2]
    alto_contenido = max(alto_v, ALTO_MIN)
    lienzo = np.zeros((alto_contenido + ALTO_BARRA, ancho_v + ANCHO_PANEL, 3), np.uint8)
    lienzo[:] = ui.FONDO_PANEL

    vista = frame  # `frame` ya viene con el overlay de calor aplicado
    desplazamiento_y = (alto_contenido - alto_v) // 2
    lienzo[desplazamiento_y:desplazamiento_y + alto_v, :ancho_v] = vista

    textos = []
    sub = lienzo[desplazamiento_y:desplazamiento_y + alto_v, :ancho_v]
    if not detecciones:
        dibujar_sin_caras(sub, textos)
    for det, edad in zip(detecciones, edades):
        dibujar_cara(sub, det.caja, edad, textos, principal=(det is principal))
    if estado.ayuda:
        dibujar_ayuda(sub, textos)
    # Los textos del video se dibujaron con coordenadas de `sub`: los corro.
    textos = [t._replace(y=t.y + desplazamiento_y) for t in textos]

    dibujar_panel(lienzo, ancho_v, alto_contenido, expl, edad_p, dispersion_p, estado, textos)
    dibujar_barra_inferior(lienzo, alto_contenido, estado, textos)
    ui.dibujar_textos(lienzo, textos)
    return lienzo


# ----------------------------------------------------------------------------
# Pipeline por frame
# ----------------------------------------------------------------------------
def procesar(frame, models, detector, explicador, smoother, estado, margen,
             suavizador_lm=None):
    """Detecta, predice, explica y devuelve el lienzo listo para mostrar."""
    # Las caras chicas (gente al fondo) no tienen detalle suficiente: la edad
    # sale cualquier cosa y encima cada una cuesta un forward por modelo.
    detecciones = [d for d in detector.detect(frame) if d.caja[2] >= estado.cara_min]
    if estado.suavizado and suavizador_lm is not None:
        suavizador_lm.update(detecciones)
    recortes, validas = [], []
    for det in detecciones:
        recorte = obtener_recorte(frame, det, margen, estado.recorte)
        if recorte is None:
            continue
        validas.append(det)
        recortes.append(recorte)

    # En vivo alcanza con el recorte central: con el suavizado temporal prendido,
    # 5 recortes por frame solo bajan el rango de la edad mostrada de ~1.2 a
    # ~0.9 anios y cuestan ~20 ms por frame. Se suben con --recortes-tta.
    variantes = VARIANTES_TTA[:estado.recortes_tta]
    crudas, dispersiones = [], []
    for recorte in recortes:
        edad, dispersion = predecir_edad(models, variantes_recorte(frame, recorte, variantes))
        crudas.append(edad)
        dispersiones.append(dispersion)

    if estado.suavizado and validas:
        edades = [edad for _, edad in smoother.update(
            [(d.caja, e) for d, e in zip(validas, crudas)])]
    else:
        edades = [clamp_age(e) for e in crudas]

    principal, expl, edad_p, dispersion_p = None, None, None, None
    if validas:
        indice = int(np.argmax([d.area for d in validas]))
        principal = validas[indice]
        edad_p = edades[indice]
        dispersion_p = dispersiones[indice]
        if estado.explicar:
            expl = explicador.explicar(recortes[indice], principal.landmarks,
                                       refrescar=not estado.congelado,
                                       suavizar=estado.suavizado)
            if estado.overlay:
                superponer_en_frame(frame, recortes[indice], expl.mapa)
        else:
            expl = explicador.solo_miniatura(recortes[indice])

    return componer(frame, validas, edades, principal, expl, edad_p, dispersion_p, estado)


# ----------------------------------------------------------------------------
# Modos
# ----------------------------------------------------------------------------
def calentar(models, explicador):
    """Primera pasada en frio, con una imagen ficticia.

    En MPS la primera inferencia tarda ~600 ms mientras se compilan los kernels,
    y lo mismo cada metodo de explicabilidad la primera vez. Conviene pagarlo al
    arrancar y no en el primer frame con gente mirando.
    """
    ficticia = np.full((256, 256, 3), 128, np.uint8)
    recorte = Recorte(ficticia, np.eye(2, 3, dtype=np.float32), 256)
    predecir_edad(models, ficticia)
    metodo_original = explicador.metodo
    for metodo in ('gradcam', 'oclusion'):
        explicador.metodo = metodo
        explicador.explicar(recorte)
    explicador.metodo = metodo_original
    explicador.invalidar()


def guardar_captura(lienzo, frame_crudo=None):
    """Guarda la pantalla y, si se pasa, el frame de la camara sin nada dibujado
    (`..._crudo.png`), que es el que sirve para volver a correrlo con --image."""
    os.makedirs(CARPETA_CAPTURAS, exist_ok=True)
    base = os.path.join(CARPETA_CAPTURAS, f"demo_{dt.datetime.now():%Y%m%d_%H%M%S}")
    cv2.imwrite(base + '.png', lienzo)
    if frame_crudo is not None:
        cv2.imwrite(base + '_crudo.png', frame_crudo)
    print(f"[ok] Captura guardada en {base}.png")
    return base + '.png'


def run_image(models, detector, explicador, image_path, margen, estado):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[x] No pude abrir la imagen: {image_path}")
        return
    estado.suavizado = False
    estado.fps = None

    # Se detecta e informa ANTES de componer, porque `procesar` pinta el mapa de
    # calor sobre `frame` y detectar de nuevo ahi daria otra cosa.
    caras = [d for d in detector.detect(frame) if d.caja[2] >= estado.cara_min]
    if not caras:
        print("[i] No detecte ninguna cara en la imagen.")
    for det in caras:
        recorte = obtener_recorte(frame, det, margen, estado.recorte)
        if recorte is None:
            continue
        edad, dispersion = predecir_edad(models, variantes_recorte(frame, recorte))
        print(f"    Cara en {det.caja} -> {clamp_age(edad)} anios "
              f"(dispersion entre vistas: {dispersion:.1f})")

    lienzo = procesar(frame, models, detector, explicador, AgeSmoother(), estado, margen)
    os.makedirs(CARPETA_CAPTURAS, exist_ok=True)
    salida = os.path.join(CARPETA_CAPTURAS, 'demo_output.jpg')
    cv2.imwrite(salida, lienzo)
    print(f"[ok] Guarde el resultado en {salida}")


TITULO_VENTANA = "Estimación de edad — Semana de la Salud Mental"


def poner_titulo_unicode(nombre_ventana, titulo=TITULO_VENTANA):
    """En Windows, cambia el titulo visible de la ventana por uno con acentos.

    OpenCV identifica la ventana por `nombre_ventana` (que tiene que ser ASCII)
    en su propia tabla, no por el texto de la barra, asi que cambiar el texto
    con la API Unicode de Windows no rompe imshow ni waitKey. En Linux y macOS
    el backend ya maneja UTF-8 y no hace falta.
    """
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, nombre_ventana)
        if hwnd:
            ctypes.windll.user32.SetWindowTextW(hwnd, titulo)
    except Exception:
        pass


def tamano_pantalla():
    """(ancho, alto) de la pantalla, o None si no se puede saber.

    OpenCV no lo expone; tkinter viene con Python en Windows, macOS y la
    mayoria de los Linux. En Windows con escalado de pantalla (125%, 150%...)
    devuelve el tamano en las mismas unidades que usa la ventana de OpenCV, asi
    que el ajuste de abajo sale bien igual.
    """
    try:
        import tkinter
        raiz = tkinter.Tk()
        raiz.withdraw()
        tamano = raiz.winfo_screenwidth(), raiz.winfo_screenheight()
        raiz.destroy()
        return tamano
    except Exception:
        return None


def ajustar_ventana(titulo, lienzo, pantalla, fraccion=0.9):
    """Agranda (o achica) la ventana para que ocupe ~90% de la pantalla sin
    deformar el lienzo. El lienzo se dibuja en pixeles fijos y OpenCV lo
    muestra 1:1, asi que en un monitor grande o de alta resolucion todo se veia
    chiquito; con la ventana WINDOW_NORMAL el sistema lo escala entero."""
    if pantalla is None:
        return
    ancho_p, alto_p = pantalla
    # Con dos monitores lado a lado tkinter devuelve el escritorio entero (p.ej.
    # 3840x1080) y la ventana quedaria partida entre los dos: se asume un
    # monitor de 16:9 como maximo.
    ancho_p = min(ancho_p, alto_p * 16 // 9)
    alto_l, ancho_l = lienzo.shape[:2]
    factor = fraccion * min(ancho_p / ancho_l, alto_p / alto_l)
    cv2.resizeWindow(titulo, int(ancho_l * factor), int(alto_l * factor))


def run_webcam(models, detector, explicador, camera, margen, estado, smoother,
               ajustar=True):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"[x] No pude abrir la camara {camera}.")
        return
    # Muchas webcams USB dan 1280x720 a 30 fps solo en MJPG; en el formato por
    # defecto (YUYV) bajan a 10-15 fps y la demo no puede ir mas rapido que eso.
    # Si la camara no soporta MJPG, OpenCV ignora el pedido.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"[i] Camara: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} a {cap.get(cv2.CAP_PROP_FPS):.0f} fps")

    # Solo ASCII: en Windows OpenCV le pasa el titulo al sistema como texto
    # ANSI y cualquier caracter fuera de ASCII (acentos, la raya) sale como
    # basura. `poner_titulo_unicode` lo reemplaza por la version con acentos.
    titulo = "Estimacion de edad - Semana de la Salud Mental"
    pantalla = tamano_pantalla() if ajustar else None
    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    print("[i] Webcam abierta. Apreta 'h' en la ventana para ver las teclas.")
    ventana_ajustada = False

    suavizador_lm = SuavizadorLandmarks()
    ultimo, congelado = time.time(), None
    lienzo = None

    while True:
        if estado.congelado and congelado is not None:
            frame = congelado.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # espejo, mas natural
            congelado = frame.copy()

        lienzo = procesar(frame, models, detector, explicador, smoother, estado, margen,
                          suavizador_lm)

        ahora = time.time()
        estado.fps = 0.9 * estado.fps + 0.1 * (1.0 / max(ahora - ultimo, 1e-6))
        ultimo = ahora

        cv2.imshow(titulo, lienzo)
        if not ventana_ajustada:
            ajustar_ventana(titulo, lienzo, pantalla)
            poner_titulo_unicode(titulo)
            ventana_ajustada = True
        # Mientras esta congelado no hay que correr a full: el frame no cambia.
        tecla = cv2.waitKey(30 if estado.congelado else 1) & 0xFF
        if tecla in (ord('q'), 27):
            break
        elif tecla == ord(' '):
            estado.congelado = not estado.congelado
        elif tecla == ord('s'):
            estado.suavizado = not estado.suavizado
        elif tecla == ord('e'):
            estado.explicar = not estado.explicar
        elif tecla == ord('o'):
            estado.overlay = not estado.overlay
        elif tecla == ord('r'):
            estado.contornos = not estado.contornos
        elif tecla == ord('h'):
            estado.ayuda = not estado.ayuda
        elif tecla == ord('m'):
            explicador.alternar_metodo()
        elif tecla == ord('l'):
            explicador.alternar_capa()
        elif tecla == ord('g') and lienzo is not None:
            guardar_captura(lienzo, congelado)
        elif tecla == ord('f'):
            estado.pantalla_completa = not estado.pantalla_completa
            cv2.setWindowProperty(
                titulo, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if estado.pantalla_completa else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
def main():
    global SIGMA_ENTRADA, REDUCIR_ENTRADA, AUTOCONTRASTE, DEVICE
    ap = argparse.ArgumentParser(description="Demo en vivo de estimacion de edad.")
    ap.add_argument('--modelo', default='combinado', choices=['combinado', 'arista', 'resnet101'],
                    help="combinado: promedio de resnet101 y arista (el de menor error en test). "
                         "arista: ResNet-50 a 128 px. resnet101: ResNet-101 a 200 px con "
                         "cabeza de distribucion (training/train_resnet101.py).")
    ap.add_argument('--model', default=os.path.join(RAIZ, 'models', 'arista.pt'),
                    help="Pesos de arista.")
    ap.add_argument('--pesos-resnet101', default=os.path.join(RAIZ, 'models', 'resnet101_robusta.pt'),
                    help="Pesos de la ResNet-101 (por defecto la entrenada con --aug fuerte, "
                         "mas estable en webcam; la otra es models/resnet101.pt).")
    ap.add_argument('--ensemble', action='store_true',
                    help="Promediar arista + finetuneada (baja un poco el error).")
    ap.add_argument('--model2', default=os.path.join(RAIZ, 'models', 'fine-tuneada.pt'),
                    help="Pesos del 2do modelo (finetuneada) para el ensemble.")
    ap.add_argument('--image', default=None,
                    help="Correr sobre una imagen en vez de la webcam.")
    ap.add_argument('--camera', type=int, default=0, help="Indice de camara.")
    ap.add_argument('--margin', type=float, default=0.2,
                    help="Margen alrededor de la cara al recortar.")
    ap.add_argument('--detector', default='auto', choices=['auto', 'yunet', 'haar'])
    ap.add_argument('--recortes-tta', type=int, default=1, choices=range(1, 6),
                    metavar='{1..5}',
                    help="Recortes levemente corridos que se promedian por frame. 1 = mas "
                         "fluido; 5 = un poco mas estable si la GPU lo banca.")
    ap.add_argument('--cara-min', type=int, default=120,
                    help="Ancho minimo de cara (px del frame) para estimarle la edad.")
    ap.add_argument('--blur', type=float, default=SIGMA_ENTRADA,
                    help="Sigma de un desenfoque del recorte antes de la red; alternativa "
                         "a --reducir (p.ej. --blur 1.0 --reducir 0).")
    ap.add_argument('--reducir', type=int, default=REDUCIR_ENTRADA,
                    help="Achicar el recorte a N px antes de la red, para parecerse a la "
                         "nitidez de UTKFace (0 = no achicar).")
    ap.add_argument('--sin-autocontraste', dest='autocontraste', action='store_false',
                    help="No estirar el contraste de la cara antes de la red.")
    ap.add_argument('--ancho-deteccion', type=int, default=640,
                    help="Ancho al que se achica el frame para detectar (0 = sin achicar).")
    ap.add_argument('--recorte', default='alineado', choices=['alineado', 'cuadrado'],
                    help="'alineado' reproduce el encuadre de UTKFace con los landmarks "
                         "(bastante mas preciso); 'cuadrado' es el recorte con margen de "
                         "antes. Sin landmarks (Haar) cae siempre a cuadrado.")
    ap.add_argument('--no-smooth', dest='smooth', action='store_false',
                    help="Arrancar sin suavizado temporal: edad, mapa y barras crudos "
                         "frame a frame (inestable, pero muestra como es la red en realidad).")
    ap.add_argument('--suavizado', default='ema', choices=['ema', 'mediana'],
                    help="Como se suaviza la edad: media exponencial (ema) o mediana "
                         "de las ultimas --ventana predicciones (robusta a saltos aislados).")
    ap.add_argument('--ventana', type=int, default=9,
                    help="Cantidad de frames de la mediana (solo con --suavizado mediana).")
    ap.add_argument('--metodo', default='gradcam', choices=['gradcam', 'oclusion'],
                    help="Metodo de explicabilidad con el que arranca.")
    ap.add_argument('--capa', default='layer4', choices=['layer1', 'layer2', 'layer3', 'layer4'],
                    help="Capa para el Grad-CAM. layer4 (7x7 en la ResNet-101 a 200 px) da "
                         "zonas mas claras; layer3 da mas detalle pero mas disperso.")
    ap.add_argument('--grilla', type=int, default=8,
                    help="Resolucion de la grilla de oclusion (grilla^2 forwards por mapa).")
    ap.add_argument('--sin-explicacion', dest='explicar', action='store_false',
                    help="Arrancar con la explicacion apagada.")
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu', 'mps'],
                    help="Donde corren los modelos. auto = GPU si hay. En CPU conviene "
                         "--modelo arista, que es bastante mas liviano.")
    ap.add_argument('--escala-ui', type=float, default=1.0,
                    help="Tamano de toda la interfaz (letra, panel, margenes) respecto "
                         "del video. 1.5 = todo 50%% mas grande.")
    ap.add_argument('--no-ajustar-ventana', dest='ajustar_ventana', action='store_false',
                    help="No agrandar la ventana al ~90%% de la pantalla al arrancar.")
    ap.add_argument('--overlay', action='store_true',
                    help="Arrancar pintando el mapa de calor tambien sobre el video "
                         "(por defecto va solo en el panel de la derecha).")
    args = ap.parse_args()
    # Antes de crear el Explicador: TAM_EXPL (el lado del mapa) depende de la escala.
    aplicar_escala_ui(args.escala_ui)

    rutas = {'arista': args.model, 'resnet101': args.pesos_resnet101}
    usados = ['resnet101', 'arista'] if args.modelo == 'combinado' else [args.modelo]
    for nombre in usados:
        if not os.path.exists(rutas[nombre]):
            print(f"[x] No encuentro los pesos de {nombre}: {rutas[nombre]}")
            print("    Pone los .pt entrenados en models/ (o pasa la ruta por argumento).")
            sys.exit(1)

    if args.device != 'auto':
        DEVICE = args.device
    print(f"[i] Device: {DEVICE}")
    # El primero es el que se explica: la ResNet-101 a 200 px da mapas que
    # cambian mucho mas entre caras que arista a 128 px.
    cargar = {'arista': load_arista, 'resnet101': load_resnet101}
    models = [cargar[nombre](rutas[nombre]) for nombre in usados]
    print(f"[i] Modelo: {' + '.join(usados)}")
    estado = Estado(suavizado=args.smooth, explicar=args.explicar,
                    recorte=args.recorte, overlay=args.overlay,
                    recortes_tta=args.recortes_tta, cara_min=args.cara_min)
    SIGMA_ENTRADA = args.blur
    REDUCIR_ENTRADA = args.reducir
    AUTOCONTRASTE = args.autocontraste
    estado.ensemble = len(models) > 1
    if args.ensemble:
        if os.path.exists(args.model2):
            models.append(load_finetuneada(args.model2))
            estado.ensemble = True
            print("[i] Ensemble activo: arista + finetuneada")
        else:
            print(f"[!] No encontre {args.model2}; sigo solo con arista.")

    detector = FaceDetector(prefer=args.detector, ancho_max=args.ancho_deteccion)
    if estado.recorte == 'alineado' and not detector.da_landmarks:
        print("[!] El detector Haar no da landmarks: uso el recorte cuadrado.")
        print("    Para el recorte alineado hace falta YuNet "
              f"({YUNET_PATH}), que predice bastante mejor.")
        estado.recorte = 'cuadrado'
    # La explicacion describe SIEMPRE al primer modelo: un mapa promediado entre
    # dos redes distintas no explicaria a ninguna de las dos.
    explicador = Explicador(models[0], capa=args.capa, metodo=args.metodo, grilla=args.grilla)
    calentar(models, explicador)

    try:
        if args.image:
            run_image(models, detector, explicador, args.image, args.margin, estado)
        else:
            smoother = AgeSmoother(modo=args.suavizado, ventana=args.ventana)
            run_webcam(models, detector, explicador, args.camera, args.margin, estado,
                       smoother, ajustar=args.ajustar_ventana)
    finally:
        explicador.cerrar()


if __name__ == '__main__':
    main()
