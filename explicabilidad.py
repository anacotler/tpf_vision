"""
Explicabilidad para la demo de estimacion de edad.

Responde "que parte de la cara esta mirando la red" de dos formas distintas,
que en el stand conviene mostrar una despues de la otra porque dicen cosas
distintas:

  Grad-CAM (barato, ~1 forward + 1 backward)
      Pesa los mapas de activacion de una capa convolucional por el gradiente
      de la edad predicha respecto de esa capa. Contesta "que zonas usa la red
      para armar el numero". Es correlacional: sale del gradiente, no de una
      intervencion. Resolucion baja (layer4 -> 4x4, layer3 -> 8x8 con entrada
      de 128px), asi que el mapa se ve como manchas suaves.

  Oclusion (caro, grilla*grilla forwards)
      Tapa un parche de la cara y mide cuanto se mueve la edad predicha.
      Contesta "que pasa si le saco esto", que es causal y ademas se explica
      solo: "si le tapo los ojos dice 8 anios mas". Se paga con ~64 forwards
      por mapa, por eso se calcula con menos frecuencia.

Ademas reparte el mapa en regiones anatomicas (ojos, nariz, boca, frente,
mejillas y menton) para poder decir en texto donde esta mirando, que es lo que
la gente entiende sin saber que es un mapa de calor.
"""
import math
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F

NOMBRES_REGIONES = ('Ojos', 'Nariz', 'Boca', 'Frente', 'Mejillas y mentón')

# Indices de las capas de ResNet dentro del nn.Sequential que arma model_arista.
_INDICES_BACKBONE = {'layer1': 4, 'layer2': 5, 'layer3': 6, 'layer4': 7}


# ----------------------------------------------------------------------------
# Paleta del mapa de calor
# ----------------------------------------------------------------------------
# Rampa secuencial de UN SOLO tono, claro -> oscuro. Dos razones para no usar
# una paleta tipo INFERNO/JET aunque sean las tipicas de los papers:
#
#   1. Una rampa secuencial correcta es de un solo tono; las multitono (arcoiris)
#      inventan fronteras donde el dato es continuo.
#   2. Esto se compone ENCIMA DE UNA CARA. El extremo bajo de INFERNO es violeta
#      oscuro y, con alpha bajo sobre piel, tine toda la cara de violeta: se lee
#      como un moreton. Sobre una persona real eso no es aceptable. El cian no
#      existe como color de piel ni de lesion, asi que se lee siempre como una
#      capa encima y nunca como algo que le pasa a la cara.
#
# El extremo claro queda casi transparente (el alpha es proporcional al valor),
# asi que la cara fuera del foco queda intacta. Luminosidad monotona decreciente
# en OKLab (L de 0.95 a 0.52, sin mesetas), que es el chequeo que corresponde a
# una rampa secuencial.
_PASOS_CALOR = ('#d4f5f7', '#93e4ea', '#3fc9d6', '#12a3b5', '#0b7686')


def _lut_de_rampa(pasos):
    """Pasos hex -> LUT 256x1x3 BGR para cv2.applyColorMap."""
    puntos = np.array([[int(h[i:i + 2], 16) for i in (5, 3, 1)] for h in pasos],
                      dtype=np.float32)          # hex es RGB, la LUT va en BGR
    posiciones = np.linspace(0, 255, len(puntos))
    rampa = np.stack([np.interp(np.arange(256), posiciones, puntos[:, c])
                      for c in range(3)], axis=1)
    return rampa.astype(np.uint8).reshape(256, 1, 3)


PALETA_CALOR = _lut_de_rampa(_PASOS_CALOR)


# ----------------------------------------------------------------------------
# Utilidades de mapas
# ----------------------------------------------------------------------------
def normalizar(mapa):
    """Lleva un mapa a [0, 1]. Si es constante, devuelve ceros."""
    mapa = np.asarray(mapa, dtype=np.float32)
    minimo = float(mapa.min())
    mapa = mapa - minimo
    maximo = float(mapa.max())
    if maximo < 1e-8:
        return np.zeros_like(mapa)
    return mapa / maximo


def colorear(mapa, paleta=PALETA_CALOR):
    """Mapa en [0, 1] -> imagen BGR con la paleta elegida."""
    return cv2.applyColorMap((np.clip(mapa, 0, 1) * 255).astype(np.uint8), paleta)


def superponer(imagen_bgr, mapa, intensidad=0.80, gamma=1.6, paleta=PALETA_CALOR):
    """Pinta el mapa sobre la imagen con alpha proporcional al calor.

    El alpha variable (en vez de uno fijo para toda la imagen) es lo que hace
    que las zonas frias queden intactas y solo se tinte lo que la red mira.

    `gamma` > 1 apaga todavia mas las zonas tibias. Es puro ajuste de
    visualizacion: sin el, un mapa normalizado deja ~0.3 de calor en todos
    lados y la cara entera queda con un velo violeta. Al ser una transformacion
    monotona no cambia el orden de lo que se ve mas caliente, y los porcentajes
    por region se calculan sobre el mapa crudo, no sobre este.
    """
    alto, ancho = imagen_bgr.shape[:2]
    if mapa.shape[:2] != (alto, ancho):
        mapa = cv2.resize(mapa, (ancho, alto), interpolation=cv2.INTER_CUBIC)
    mapa = np.clip(mapa, 0, 1) ** gamma
    calor = colorear(mapa, paleta).astype(np.float32)
    alpha = (mapa * intensidad)[..., None]
    mezcla = imagen_bgr.astype(np.float32) * (1 - alpha) + calor * alpha
    return np.clip(mezcla, 0, 255).astype(np.uint8)


@lru_cache(maxsize=8)
def ventana_ovalo(lado):
    """Ovalo suave (1 en el centro, 0 en las esquinas) del tamano del recorte.

    El recorte es cuadrado y tiene margen, asi que pintar el mapa entero mancha
    tambien el fondo alrededor de la cara. Multiplicando por esto, el calor
    queda contenido en la cara.
    """
    ejes_y, ejes_x = np.mgrid[0:lado, 0:lado].astype(np.float32)
    centro = (lado - 1) / 2.0
    u = (ejes_x - centro) / (0.44 * lado)
    v = (ejes_y - centro) / (0.50 * lado)
    radio = np.sqrt(u * u + v * v)
    return np.clip((1.15 - radio) / 0.30, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------------
def resolver_capa(modelo, nombre='layer4'):
    """Devuelve el modulo de la capa pedida, sirviendo las dos familias de
    modelos del proyecto: las que exponen `backbone` (model_arista, model2) y
    las que envuelven un torchvision resnet en `self.model` (el resto)."""
    if nombre not in _INDICES_BACKBONE:
        raise ValueError(f"Capa desconocida: {nombre}. Use una de {list(_INDICES_BACKBONE)}")
    if hasattr(modelo, 'backbone'):
        return modelo.backbone[_INDICES_BACKBONE[nombre]]
    interno = getattr(modelo, 'model', modelo)
    if hasattr(interno, nombre):
        return getattr(interno, nombre)
    raise ValueError(f"No supe encontrar {nombre} en {type(modelo).__name__}")


class GradCAM:
    """Grad-CAM sobre una capa convolucional, para una salida de regresion.

    Ojo con el signo: en clasificacion se aplica ReLU y se lee "que apoya a esta
    clase". Aca la salida es un escalar (la edad), asi que:
      modo='sensibilidad' (default) -> |contribucion|, o sea "de que zonas
          depende el numero", que es lo que queremos mostrar.
      modo='sube_edad' -> ReLU, el Grad-CAM clasico: solo las zonas que empujan
          la edad hacia arriba.
    """

    def __init__(self, modelo, capa, modo='sensibilidad'):
        self.modelo = modelo
        self.capa = capa
        self.modo = modo
        self._activaciones = None
        self._handle = capa.register_forward_hook(self._guardar_activaciones)

    def _guardar_activaciones(self, modulo, entrada, salida):
        self._activaciones = salida

    def cerrar(self):
        """Saca el hook. Importante si se cambia de capa en caliente, si no
        quedan hooks colgados acumulandose sobre el mismo modelo."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __call__(self, lote, normalizar_salida=True):
        """lote: [B, 3, H, W] ya normalizado. Devuelve (predicciones, mapas).

        predicciones: tensor [B] en anios.
        mapas: ndarray [B, H, W] llevado al tamano de la entrada.

        `normalizar_salida=False` devuelve el mapa crudo (no negativo, sin
        min-max). Hace falta para suavizarlo en el tiempo: el min-max es por
        frame, asi que un pico espurio en un solo frame reescala todo el mapa y
        se ve como un parpadeo. Quien muestre el mapa de a uno puede dejarlo en
        True y olvidarse.
        """
        alto, ancho = lote.shape[-2:]
        self._activaciones = None
        # requires_grad en la entrada asegura que se arme el grafo aunque el
        # modelo tenga los parametros congelados.
        lote = lote.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            salida = self.modelo(lote)                       # [B, 1]
            if self._activaciones is None:
                raise RuntimeError("El hook no capturo activaciones; revisa la capa elegida.")
            # grad explicito respecto de las activaciones: no toca los .grad de
            # los parametros, asi que no hay que limpiar nada entre frames.
            gradientes = torch.autograd.grad(salida.sum(), self._activaciones)[0]

        activaciones = self._activaciones.detach()
        pesos = gradientes.mean(dim=(2, 3), keepdim=True)     # GAP del gradiente
        mapa = (pesos * activaciones).sum(dim=1, keepdim=True)  # [B, 1, h, w]
        mapa = mapa.abs() if self.modo == 'sensibilidad' else F.relu(mapa)
        mapa = F.interpolate(mapa, size=(alto, ancho), mode='bilinear', align_corners=False)

        mapas = mapa.squeeze(1).detach().cpu().numpy()
        if normalizar_salida:
            mapas = np.stack([normalizar(m) for m in mapas])
        return salida.detach().squeeze(-1), mapas


# ----------------------------------------------------------------------------
# Oclusion
# ----------------------------------------------------------------------------
@torch.no_grad()
def mapa_oclusion(modelo, tensor, grilla=8, tam_parche=0.34, lote_max=32,
                  normalizar_salida=True):
    """Tapa parches y mide cuanto se mueve la edad predicha.

    tensor: [1, 3, H, W] normalizado. Devuelve (edad_base, mapa, delta_max):
      edad_base: prediccion sin tapar nada, en anios.
      mapa: [H, W]. Normalizado a [0, 1], o crudo en anios de cambio si
          normalizar_salida=False (ver la nota en GradCAM.__call__).
      delta_max: cuantos anios se movio la prediccion en el peor parche. Sirve
          para decirlo en palabras ("taparle los ojos le suma 6 anios").

    El parche se rellena con una version borroneada de la propia imagen en vez
    de gris plano: gris plano mete un rectangulo que la red nunca vio en
    entrenamiento y el mapa termina midiendo "que raro es el parche" mas que
    "cuanto importaba esa zona".
    """
    alto, ancho = tensor.shape[-2:]
    fondo = F.avg_pool2d(tensor, kernel_size=21, stride=1, padding=10)
    edad_base = float(modelo(tensor).reshape(-1)[0].item())

    lado = max(2, int(round(tam_parche * min(alto, ancho))))
    centros_y = ((np.arange(grilla) + 0.5) / grilla * alto).astype(int)
    centros_x = ((np.arange(grilla) + 0.5) / grilla * ancho).astype(int)

    variantes = []
    for cy in centros_y:
        for cx in centros_x:
            y0, y1 = max(0, cy - lado // 2), min(alto, cy + lado // 2)
            x0, x1 = max(0, cx - lado // 2), min(ancho, cx + lado // 2)
            variante = tensor.clone()
            variante[:, :, y0:y1, x0:x1] = fondo[:, :, y0:y1, x0:x1]
            variantes.append(variante)

    predicciones = []
    for inicio in range(0, len(variantes), lote_max):
        lote = torch.cat(variantes[inicio:inicio + lote_max], dim=0)
        predicciones.append(modelo(lote).reshape(-1).float().cpu())
    predicciones = torch.cat(predicciones).numpy()

    delta = np.abs(predicciones - edad_base).reshape(grilla, grilla)
    delta_max = float(delta.max())
    mapa = cv2.resize(delta.astype(np.float32), (ancho, alto), interpolation=cv2.INTER_CUBIC)
    return edad_base, (normalizar(mapa) if normalizar_salida else np.maximum(mapa, 0)), delta_max


# ----------------------------------------------------------------------------
# Regiones anatomicas
# ----------------------------------------------------------------------------
def landmarks_estimados(caja_en_recorte):
    """5 puntos (ojo der, ojo izq, nariz, comisura der, comisura izq) a partir
    de proporciones tipicas de una cara frontal. Es el plan B para cuando el
    detector no da landmarks (Haar); con YuNet usamos los de verdad."""
    x, y, ancho, alto = caja_en_recorte
    punto = lambda fx, fy: (x + fx * ancho, y + fy * alto)
    return np.array([
        punto(0.31, 0.40), punto(0.69, 0.40),   # ojos
        punto(0.50, 0.60),                      # nariz
        punto(0.37, 0.77), punto(0.63, 0.77),   # comisuras
    ], dtype=np.float32)


def _elipse(tam, centro, radio_x, radio_y, angulo):
    mascara = np.zeros((tam, tam), dtype=np.float32)
    cv2.ellipse(
        mascara,
        (int(round(centro[0])), int(round(centro[1]))),
        (max(1, int(round(radio_x))), max(1, int(round(radio_y)))),
        angulo, 0, 360, 1.0, -1,
    )
    # Bordes blandos: las regiones de la cara no tienen limites nitidos y sin
    # esto la atribucion salta feo cuando la cabeza se mueve un pixel.
    return cv2.GaussianBlur(mascara, (0, 0), max(1.0, tam * 0.02))


def mascaras_regiones(landmarks, tam):
    """Devuelve dict nombre -> mascara float32 [tam, tam] en [0, 1].

    Todo se mide en unidades de `d` = distancia entre ojos, y se orienta con la
    linea de los ojos, asi que aguanta que la persona incline la cabeza.
    """
    landmarks = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    ojo_der, ojo_izq, nariz, com_der, com_izq = landmarks
    centro_ojos = (ojo_der + ojo_izq) / 2.0
    centro_boca = (com_der + com_izq) / 2.0

    eje = ojo_izq - ojo_der
    distancia = float(np.linalg.norm(eje))
    if distancia < 1e-3:
        # Landmarks degenerados: caigo a una cara frontal centrada.
        return mascaras_regiones(landmarks_estimados((0.15 * tam, 0.15 * tam,
                                                      0.70 * tam, 0.70 * tam)), tam)
    horizontal = eje / distancia
    angulo = math.degrees(math.atan2(float(horizontal[1]), float(horizontal[0])))
    # Perpendicular, orientada hacia la boca (o sea, "hacia abajo" en la cara).
    vertical = np.array([-horizontal[1], horizontal[0]], dtype=np.float32)
    if float(np.dot(vertical, centro_boca - centro_ojos)) < 0:
        vertical = -vertical

    regiones = {
        'Ojos':   _elipse(tam, centro_ojos, 1.05 * distancia, 0.42 * distancia, angulo),
        'Nariz':  _elipse(tam, nariz, 0.42 * distancia, 0.50 * distancia, angulo),
        'Boca':   _elipse(tam, centro_boca, 0.78 * distancia, 0.40 * distancia, angulo),
        'Frente': _elipse(tam, centro_ojos - 0.85 * distancia * vertical,
                          1.05 * distancia, 0.52 * distancia, angulo),
    }
    # El resto de la cara (mejillas, mandibula, menton) es el ovalo facial menos
    # lo ya asignado, para que las cinco regiones cubran la cara sin pisarse.
    ovalo = _elipse(tam, centro_ojos + 0.55 * distancia * vertical,
                    1.45 * distancia, 2.00 * distancia, angulo)
    ocupado = np.clip(sum(regiones.values()), 0, 1)
    regiones['Mejillas y mentón'] = np.clip(ovalo - ocupado, 0, 1)
    return regiones


def atribucion_por_region(mapa, mascaras):
    """Reparte el mapa entre regiones y devuelve [(nombre, porcentaje), ...]
    ordenado de mayor a menor.

    Usa la intensidad *media* dentro de cada region, no la suma: con la suma
    'Mejillas y menton' ganaria siempre por ser la region mas grande.
    """
    crudos = {}
    for nombre, mascara in mascaras.items():
        peso = float(mascara.sum())
        crudos[nombre] = float((mapa * mascara).sum() / peso) if peso > 1e-6 else 0.0
    total = sum(crudos.values())
    if total <= 1e-8:
        return [(nombre, 0.0) for nombre in NOMBRES_REGIONES]
    return sorted(((n, 100.0 * v / total) for n, v in crudos.items()),
                  key=lambda par: -par[1])


def dibujar_contornos_regiones(imagen_bgr, mascaras, color=(200, 200, 200), grosor=1):
    """Contornos tenues de las regiones, para que se vea que el reparto en
    texto no es inventado sino que corresponde a estas zonas."""
    salida = imagen_bgr.copy()
    for mascara in mascaras.values():
        binaria = (mascara > 0.5).astype(np.uint8)
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(salida, contornos, -1, color, grosor, cv2.LINE_AA)
    return salida
