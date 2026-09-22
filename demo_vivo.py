"""
Demo en vivo de estimacion de edad — Semana de la Salud Mental (UdeSA).

Toma la webcam, detecta caras, muestra la edad estimada en tiempo real y, en un
panel al costado, *que parte de la cara* esta mirando la red para decidir ese
numero (Grad-CAM u oclusion, ver explicabilidad.py).

El recorte por defecto es ALINEADO: usa los 5 landmarks de YuNet para reproducir
el encuadre de UTKFace (ver PLANTILLA_UTK). Importa mucho mas de lo que parece
—el modelo se entreno con caras alineadas y es muy sensible al encuadre—; sobre
img_test/ el error medio baja de 9.6 a 1.6 anios, y una cara de 73 pasa de
predecirse 33 a 71. Con --recorte cuadrado se vuelve al recorte con margen.

Uso tipico (con la webcam):
    python demo_vivo.py

Probar sobre una imagen (sin webcam), guarda la ventana entera a disco:
    python demo_vivo.py --image fotosprueba/1_0_0_20161219204512212.jpg.chip.jpg

Bajar el error con ensemble de los 2 mejores modelos:
    python demo_vivo.py --ensemble

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
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import interfaz as ui
from explicabilidad import (GradCAM, atribucion_por_region, dibujar_contornos_regiones,
                            landmarks_estimados, mapa_oclusion, mascaras_regiones,
                            normalizar, resolver_capa, superponer, ventana_ovalo)
from model_arista import AgeEstimationModel as AristaModel

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

# Modelo YuNet (opcional pero muy recomendado: ademas de detectar mejor, da los
# 5 landmarks que usamos para nombrar las regiones de la cara).
YUNET_PATH = os.path.join('models', 'face_detection_yunet_2023mar.onnx')

ANCHO_PANEL = 400          # ancho del panel de explicacion, a la derecha
ALTO_BARRA = 58            # franja inferior con el disclaimer
ALTO_MIN = 730             # alto minimo del lienzo, para que entre el panel
TAM_EXPL = 260             # lado en el que se calcula y se muestra el mapa

CARPETA_CAPTURAS = 'capturas'

# Plantilla de alineacion de UTKFace, en fracciones del lado del recorte.
# UTKFace no son fotos crudas: son "chips" ya alineados, y por eso los 5 puntos
# caen casi siempre en el mismo lugar. Estos valores salen de correr YuNet sobre
# los chips de img_test/ (desvio de 0.012-0.019 del lado en los ojos).
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

_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

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
    ("g", "guardar captura en capturas/"),
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

    def __init__(self, prefer='auto'):
        self.mode = None
        self.yunet = None
        self.haar = None

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
            self.yunet.setInputSize((ancho, alto))
            _, caras = self.yunet.detect(frame_bgr)
            if caras is None:
                return []
            salida = []
            for cara in caras:
                # Formato YuNet: x, y, w, h, 5 landmarks (x,y), score.
                # Landmarks: ojo der, ojo izq, nariz, comisura der, comisura izq.
                caja = tuple(int(v) for v in cara[:4])
                landmarks = np.asarray(cara[4:14], dtype=np.float32).reshape(5, 2)
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


def a_tensor(recorte_bgr):
    """Recorte BGR -> tensor [1, 3, 128, 128] normalizado, en el device."""
    rgb = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB)
    return _transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)


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
    model.eval()
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


@torch.no_grad()
def predecir_edad(models, crop_bgr, tta=True):
    """Predice edad promediando modelos y, si tta, la imagen espejada.

    Devuelve (edad, dispersion). La dispersion es el desvio entre las vistas
    (original / espejada / cada modelo del ensemble). NO es un intervalo de
    confianza: solo dice cuanto se contradice el sistema consigo mismo, que en
    el stand alcanza para mostrar cuando la prediccion es fragil.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    tensors = [_transform(pil)]
    if tta:
        tensors.append(_transform(pil.transpose(Image.FLIP_LEFT_RIGHT)))
    batch = torch.stack(tensors).to(DEVICE)

    vistas = []
    for model in models:
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
class AgeSmoother:
    """Estabiliza la edad en el tiempo. Asocia cada cara con la del frame
    anterior por cercania del centro y aplica una media exponencial (EMA).
    Asi el numero deja de saltar frame a frame. Se puede apagar para mostrar
    la prediccion cruda (inestable) en el stand."""

    def __init__(self, alpha=0.25, max_dist=100, ttl=12):
        self.alpha = alpha        # cuanto pesa el frame nuevo (mas bajo = mas estable)
        self.max_dist = max_dist  # distancia max (px) para considerar "la misma cara"
        self.ttl = ttl            # frames que sobrevive una cara sin volver a verse
        self.tracks = {}          # id -> {'c': (cx, cy), 'age': float, 'miss': int}
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
                self.tracks[best] = {'c': (cx, cy), 'age': float(raw), 'miss': 0}
            else:
                t = self.tracks[best]
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
    INTERVALOS = {'gradcam': 0.0, 'oclusion': 0.15}

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
        tensor = a_tensor(recorte.imagen)
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


# ----------------------------------------------------------------------------
# Dibujo
# ----------------------------------------------------------------------------
def etiqueta_edad(edad):
    return f"{edad} año" if edad == 1 else f"{edad} años"


def dibujar_cara(frame, caja, edad, textos, principal=False):
    x, y, ancho, alto = [int(v) for v in caja]
    color = ui.VERDE if principal else ui.TEXTO_TENUE
    ui.caja_esquinas(frame, (x, y, ancho, alto), color, grosor=3 if principal else 2)

    etiqueta = etiqueta_edad(edad)
    tam = 30 if principal else 22
    ancho_txt = ui.ancho_texto(etiqueta, tam, 'bold')
    ancho_pill = ancho_txt + 28
    alto_pill = tam + 16
    # Si no entra arriba de la caja, va adentro.
    py = y - alto_pill - 8 if y - alto_pill - 8 >= 0 else y + 8
    px = x
    ui.panel_traslucido(frame, (px, py), (px + ancho_pill, py + alto_pill),
                        ui.FONDO_BARRA, alpha=0.80, radio=alto_pill // 2)
    textos.append(ui.Texto(px + ancho_pill // 2, py + alto_pill // 2, etiqueta,
                           tam, color, 'bold', 'mm'))


def dibujar_sin_caras(frame, textos):
    alto, ancho = frame.shape[:2]
    ui.panel_traslucido(frame, (ancho // 2 - 220, alto // 2 - 44),
                        (ancho // 2 + 220, alto // 2 + 44), ui.FONDO_BARRA, 0.70, 16)
    textos.append(ui.Texto(ancho // 2, alto // 2 - 12, "No veo ninguna cara",
                           26, ui.TEXTO, 'bold', 'mm'))
    textos.append(ui.Texto(ancho // 2, alto // 2 + 20, "Acercate y mirá a la cámara",
                           17, ui.TEXTO_TENUE, 'regular', 'mm'))


def dibujar_ayuda(frame, textos):
    alto, ancho = frame.shape[:2]
    alto_caja = 62 + len(AYUDA) * 30
    x0, y0 = 40, max(20, alto // 2 - alto_caja // 2)
    x1, y1 = x0 + 430, y0 + alto_caja
    ui.panel_traslucido(frame, (x0, y0), (x1, y1), ui.FONDO_BARRA, 0.88, 16)
    textos.append(ui.Texto(x0 + 24, y0 + 22, "TECLAS", 14, ui.TEXTO_TENUE, 'bold'))
    for i, (tecla, que) in enumerate(AYUDA):
        y = y0 + 50 + i * 30
        textos.append(ui.Texto(x0 + 24, y, tecla, 17, ui.ACENTO, 'bold'))
        textos.append(ui.Texto(x0 + 150, y, que, 17, ui.TEXTO, 'regular'))


def dibujar_panel(lienzo, x_panel, alto_panel, expl, edad, dispersion, estado, textos):
    """Panel derecho: la cara como la ve la red, el mapa y el reparto por region.

    Las posiciones van con un cursor `y` que baja: el contenido entra en
    ALTO_MIN px, asi que si se agrega una seccion hay que subir esa constante.
    """
    px = x_panel
    lienzo[:alto_panel, px:px + ANCHO_PANEL] = ui.FONDO_PANEL
    izq, der = px + 24, px + ANCHO_PANEL - 24

    def linea(y):
        cv2.line(lienzo, (izq, y), (der, y), ui.LINEA, 1)

    textos.append(ui.Texto(izq, 20, "LO QUE MIRA LA RED", 14, ui.TEXTO_TENUE, 'bold'))
    linea(44)

    # --- La cara tal como entra a la red (128x128 reescalado), con el mapa.
    ix, iy = px + (ANCHO_PANEL - TAM_EXPL) // 2, 56
    if expl is not None:
        vista = superponer(expl.miniatura, expl.mapa, 0.85) if expl.activa else expl.miniatura.copy()
        if estado.contornos and expl.mascaras:
            vista = dibujar_contornos_regiones(vista, expl.mascaras, (120, 120, 120), 1)
        lienzo[iy:iy + TAM_EXPL, ix:ix + TAM_EXPL] = vista
        cv2.rectangle(lienzo, (ix - 1, iy - 1), (ix + TAM_EXPL, iy + TAM_EXPL), ui.LINEA, 1)
    else:
        cv2.rectangle(lienzo, (ix, iy), (ix + TAM_EXPL, iy + TAM_EXPL), (40, 37, 34), -1)
        textos.append(ui.Texto(ix + TAM_EXPL // 2, iy + TAM_EXPL // 2,
                               "esperando una cara", 16, ui.TEXTO_TENUE, 'regular', 'mm'))

    # --- Que metodo produjo ese mapa, y cuanto tardo.
    y = iy + TAM_EXPL + 18
    if expl is None:
        textos.append(ui.Texto(izq, y, "Sin cara a la vista", 17, ui.TEXTO_TENUE, 'bold'))
        textos.append(ui.Texto(izq, y + 24, "La explicación necesita una cara detectada.",
                               13, ui.TEXTO_TENUE))
    elif not expl.activa:
        textos.append(ui.Texto(izq, y, "Explicación apagada", 17, ui.TEXTO_TENUE, 'bold'))
        textos.append(ui.Texto(izq, y + 24, "Apretá 'e' para ver dónde mira.",
                               13, ui.TEXTO_TENUE))
    else:
        if expl.metodo == 'gradcam':
            titulo, detalle = "Grad-CAM", f" · {expl.capa}"
        else:
            titulo, detalle = "Oclusión", f" · grilla {expl.grilla}×{expl.grilla}"
        textos.append(ui.Texto(izq, y, titulo + detalle, 17, ui.ACENTO, 'bold'))
        textos.append(ui.Texto(der, y + 2, f"{expl.ms:.0f} ms", 14, ui.TEXTO_TENUE,
                               'regular', 'ra'))
        for i, nota in enumerate(NOTAS_METODO[expl.metodo]):
            textos.append(ui.Texto(izq, y + 24 + i * 17, nota, 13, ui.TEXTO_TENUE))

    # --- Reparto del mapa entre las regiones de la cara.
    y += 66
    linea(y)
    y += 16
    textos.append(ui.Texto(izq, y, "DÓNDE MIRA", 14, ui.TEXTO_TENUE, 'bold'))
    y += 22
    ranking = expl.ranking if (expl is not None and expl.activa) else []
    if ranking:
        tope = max(valor for _, valor in ranking) or 1.0
        for i, (nombre, valor) in enumerate(ranking[:5]):
            fila = y + i * 28
            destacado = (i == 0)
            color = ui.ACENTO if destacado else ui.TEXTO
            peso = 'medium' if destacado else 'regular'
            textos.append(ui.Texto(izq, fila, nombre, 15, color, peso))
            ui.barra(lienzo, px + 210, fila + 6, 118, 9, valor / tope,
                     ui.ACENTO if destacado else (110, 105, 100))
            textos.append(ui.Texto(der, fila, f"{valor:.0f}%", 15, color, peso, 'ra'))
    else:
        textos.append(ui.Texto(izq, y, "—", 15, ui.TEXTO_TENUE))

    # --- El numero, que es lo que la gente vino a ver.
    y += 5 * 28 + 2
    linea(y)
    y += 16
    textos.append(ui.Texto(izq, y, "EDAD ESTIMADA", 13, ui.TEXTO_TENUE, 'bold'))
    if edad is not None:
        textos.append(ui.Texto(izq, y + 18, f"{edad}", 46, ui.TEXTO, 'bold'))
        textos.append(ui.Texto(izq + 6 + ui.ancho_texto(f"{edad}", 46, 'bold'), y + 46,
                               "año" if edad == 1 else "años", 18, ui.TEXTO_TENUE))
        if dispersion is not None:
            textos.append(ui.Texto(der, y + 30, f"±{dispersion:.1f} entre vistas",
                                   13, ui.TEXTO_TENUE, 'regular', 'ra'))
    else:
        textos.append(ui.Texto(izq, y + 18, "—", 46, ui.TEXTO_TENUE, 'bold'))

    if expl is not None and expl.activa and expl.delta_max is not None:
        textos.append(ui.Texto(
            izq, y + 78,
            f"Taparle la zona más sensible le mueve la edad {expl.delta_max:.1f} años.",
            13, ui.TEXTO_TENUE))


def dibujar_barra_inferior(lienzo, y0, estado, textos):
    ancho = lienzo.shape[1]
    lienzo[y0:, :] = ui.FONDO_BARRA
    cv2.line(lienzo, (0, y0), (ancho, y0), ui.LINEA, 1)
    textos.append(ui.Texto(
        20, y0 + 12,
        "Estimación aproximada. El modelo se entrenó con UTKFace y puede errar bastante "
        "según la luz, el ángulo, la edad y los rasgos de cada cara.",
        15, ui.TEXTO_TENUE))
    estados = [
        f"{estado.fps:.1f} fps" if estado.fps is not None else None,
        DEVICE,
        "ensemble" if estado.ensemble else "1 modelo",
        f"recorte {estado.recorte}",
        f"suavizado {'on' if estado.suavizado else 'off'}",
        "CONGELADO" if estado.congelado else None,
    ]
    textos.append(ui.Texto(20, y0 + 34, "  ·  ".join(e for e in estados if e),
                           13, ui.TEXTO_TENUE))
    textos.append(ui.Texto(ancho - 20, y0 + 34, "h: ayuda  ·  q: salir",
                           13, ui.TEXTO_TENUE, 'regular', 'ra'))


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
def procesar(frame, models, detector, explicador, smoother, estado, margen):
    """Detecta, predice, explica y devuelve el lienzo listo para mostrar."""
    detecciones = detector.detect(frame)
    recortes, validas = [], []
    for det in detecciones:
        recorte = obtener_recorte(frame, det, margen, estado.recorte)
        if recorte is None:
            continue
        validas.append(det)
        recortes.append(recorte)

    crudas, dispersiones = [], []
    for recorte in recortes:
        edad, dispersion = predecir_edad(models, recorte.imagen)
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


def guardar_captura(lienzo):
    os.makedirs(CARPETA_CAPTURAS, exist_ok=True)
    nombre = os.path.join(
        CARPETA_CAPTURAS, f"demo_{dt.datetime.now():%Y%m%d_%H%M%S}.png")
    cv2.imwrite(nombre, lienzo)
    print(f"[ok] Captura guardada en {nombre}")
    return nombre


def run_image(models, detector, explicador, image_path, margen, estado):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[x] No pude abrir la imagen: {image_path}")
        return
    estado.suavizado = False
    estado.fps = None

    # Se detecta e informa ANTES de componer, porque `procesar` pinta el mapa de
    # calor sobre `frame` y detectar de nuevo ahi daria otra cosa.
    caras = detector.detect(frame)
    if not caras:
        print("[i] No detecte ninguna cara en la imagen.")
    for det in caras:
        recorte = obtener_recorte(frame, det, margen, estado.recorte)
        if recorte is None:
            continue
        edad, dispersion = predecir_edad(models, recorte.imagen)
        print(f"    Cara en {det.caja} -> {clamp_age(edad)} anios "
              f"(dispersion entre vistas: {dispersion:.1f})")

    lienzo = procesar(frame, models, detector, explicador, AgeSmoother(), estado, margen)
    salida = 'demo_output.jpg'
    cv2.imwrite(salida, lienzo)
    print(f"[ok] Guarde el resultado en {salida}")


def run_webcam(models, detector, explicador, camera, margen, estado):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"[x] No pude abrir la camara {camera}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    titulo = "Estimacion de edad — Semana de la Salud Mental"
    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    print("[i] Webcam abierta. Apreta 'h' en la ventana para ver las teclas.")

    smoother = AgeSmoother()
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

        lienzo = procesar(frame, models, detector, explicador, smoother, estado, margen)

        ahora = time.time()
        estado.fps = 0.9 * estado.fps + 0.1 * (1.0 / max(ahora - ultimo, 1e-6))
        ultimo = ahora

        cv2.imshow(titulo, lienzo)
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
            guardar_captura(lienzo)
        elif tecla == ord('f'):
            estado.pantalla_completa = not estado.pantalla_completa
            cv2.setWindowProperty(
                titulo, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if estado.pantalla_completa else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Demo en vivo de estimacion de edad.")
    ap.add_argument('--model', default='checkpoints/arista.pt',
                    help="Pesos del modelo arista (mejor modelo).")
    ap.add_argument('--ensemble', action='store_true',
                    help="Promediar arista + finetuneada (baja un poco el error).")
    ap.add_argument('--model2', default='checkpoints/finetuneada.pt',
                    help="Pesos del 2do modelo (finetuneada) para el ensemble.")
    ap.add_argument('--image', default=None,
                    help="Correr sobre una imagen en vez de la webcam.")
    ap.add_argument('--camera', type=int, default=0, help="Indice de camara.")
    ap.add_argument('--margin', type=float, default=0.2,
                    help="Margen alrededor de la cara al recortar.")
    ap.add_argument('--detector', default='auto', choices=['auto', 'yunet', 'haar'])
    ap.add_argument('--recorte', default='alineado', choices=['alineado', 'cuadrado'],
                    help="'alineado' reproduce el encuadre de UTKFace con los landmarks "
                         "(bastante mas preciso); 'cuadrado' es el recorte con margen de "
                         "antes. Sin landmarks (Haar) cae siempre a cuadrado.")
    ap.add_argument('--no-smooth', dest='smooth', action='store_false',
                    help="Arrancar sin suavizado temporal: edad, mapa y barras crudos "
                         "frame a frame (inestable, pero muestra como es la red en realidad).")
    ap.add_argument('--metodo', default='gradcam', choices=['gradcam', 'oclusion'],
                    help="Metodo de explicabilidad con el que arranca.")
    ap.add_argument('--capa', default='layer4', choices=['layer1', 'layer2', 'layer3', 'layer4'],
                    help="Capa para el Grad-CAM. layer4 da 4x4 (manchas grandes), layer3 da 8x8.")
    ap.add_argument('--grilla', type=int, default=8,
                    help="Resolucion de la grilla de oclusion (grilla^2 forwards por mapa).")
    ap.add_argument('--sin-explicacion', dest='explicar', action='store_false',
                    help="Arrancar con la explicacion apagada.")
    ap.add_argument('--overlay', action='store_true',
                    help="Arrancar pintando el mapa de calor tambien sobre el video "
                         "(por defecto va solo en el panel de la derecha).")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"[x] No encuentro los pesos: {args.model}")
        print("    Pone los .pt entrenados en checkpoints/ (o pasa --model <ruta>).")
        sys.exit(1)

    print(f"[i] Device: {DEVICE}")
    models = [load_arista(args.model)]
    estado = Estado(suavizado=args.smooth, explicar=args.explicar,
                    recorte=args.recorte, overlay=args.overlay)
    if args.ensemble:
        if os.path.exists(args.model2):
            models.append(load_finetuneada(args.model2))
            estado.ensemble = True
            print("[i] Ensemble activo: arista + finetuneada")
        else:
            print(f"[!] No encontre {args.model2}; sigo solo con arista.")

    detector = FaceDetector(prefer=args.detector)
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
            run_webcam(models, detector, explicador, args.camera, args.margin, estado)
    finally:
        explicador.cerrar()


if __name__ == '__main__':
    main()
