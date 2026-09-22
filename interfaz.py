"""
Primitivas de dibujo para la demo en vivo.

Dos motivos para que esto no sea `cv2.putText` y listo:

1. cv2.putText no dibuja acentos ni enies (la fuente Hershey es ASCII). Por eso
   la version anterior decia "anios". Aca el texto se dibuja con PIL usando una
   TrueType del sistema, asi que "~32 años" sale bien.
2. Convertir el frame numpy -> PIL -> numpy cuesta unos milisegundos, asi que
   conviene hacerlo UNA vez por frame. De ahi que `dibujar_textos` reciba la
   lista completa de textos en lugar de haber una funcion por texto suelto.

Detalle de color: los frames son BGR y PIL cree que son RGB, pero como nunca
convertimos el espacio de color alcanza con pasarle a PIL la tupla BGR tal cual
y el byte termina en el canal correcto.
"""
import os
from collections import namedtuple
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Colores en BGR.
FONDO_PANEL = (26, 23, 21)
FONDO_BARRA = (18, 16, 15)
LINEA = (62, 57, 52)
TEXTO = (242, 240, 238)
TEXTO_TENUE = (155, 150, 145)
# Cian medio (#3fc9d6). Es el mismo tono que la paleta del mapa de calor a
# proposito: las barras del panel y el mapa muestran EL MISMO dato, asi que
# comparten color. En BGR.
ACENTO = (214, 201, 63)
VERDE = (120, 215, 130)
ROJO = (90, 90, 245)

# (ruta, indice dentro del .ttc) por peso, en orden de preferencia.
_CANDIDATOS = {
    'regular': [
        ('/System/Library/Fonts/HelveticaNeue.ttc', 0),
        ('/System/Library/Fonts/Helvetica.ttc', 0),
        ('/System/Library/Fonts/Supplemental/Arial.ttf', 0),
        ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 0),
    ],
    'medium': [
        ('/System/Library/Fonts/HelveticaNeue.ttc', 10),
        ('/System/Library/Fonts/Helvetica.ttc', 0),
        ('/System/Library/Fonts/Supplemental/Arial.ttf', 0),
        ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 0),
    ],
    'bold': [
        ('/System/Library/Fonts/HelveticaNeue.ttc', 1),
        ('/System/Library/Fonts/Helvetica.ttc', 1),
        ('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 0),
        ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 0),
    ],
}

# x, y: posicion. ancla: convencion de PIL ('la' = izquierda/arriba, 'mm' =
# centrado, 'ra' = derecha/arriba, 'ls' = izquierda/linea base).
Texto = namedtuple('Texto', 'x y texto tam color peso ancla')
Texto.__new__.__defaults__ = (18, TEXTO, 'regular', 'la')


@lru_cache(maxsize=64)
def fuente(tam, peso='regular'):
    for ruta, indice in _CANDIDATOS.get(peso, _CANDIDATOS['regular']):
        if os.path.exists(ruta):
            try:
                return ImageFont.truetype(ruta, tam, index=indice)
            except OSError:
                continue
    # Ultimo recurso: la bitmap de PIL. Fea y de tamano fijo, pero no rompe.
    return ImageFont.load_default()


def ancho_texto(texto, tam, peso='regular'):
    return int(fuente(tam, peso).getlength(texto))


def dibujar_textos(frame, textos):
    """Dibuja todos los textos en una sola pasada numpy -> PIL -> numpy."""
    if not textos:
        return frame
    imagen = Image.fromarray(frame)
    lapiz = ImageDraw.Draw(imagen)
    for t in textos:
        lapiz.text((t.x, t.y), t.texto, font=fuente(t.tam, t.peso),
                   fill=tuple(int(c) for c in t.color), anchor=t.ancla)
    frame[:] = np.asarray(imagen)
    return frame


def rect_redondeado(imagen, p0, p1, color, radio=10, grosor=-1):
    """Rectangulo con esquinas redondeadas (cv2 no trae uno)."""
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    radio = int(max(0, min(radio, (x1 - x0) // 2, (y1 - y0) // 2)))
    if radio == 0:
        cv2.rectangle(imagen, (x0, y0), (x1, y1), color, grosor, cv2.LINE_AA)
        return imagen
    if grosor < 0:
        cv2.rectangle(imagen, (x0 + radio, y0), (x1 - radio, y1), color, -1, cv2.LINE_AA)
        cv2.rectangle(imagen, (x0, y0 + radio), (x1, y1 - radio), color, -1, cv2.LINE_AA)
    else:
        cv2.line(imagen, (x0 + radio, y0), (x1 - radio, y0), color, grosor, cv2.LINE_AA)
        cv2.line(imagen, (x0 + radio, y1), (x1 - radio, y1), color, grosor, cv2.LINE_AA)
        cv2.line(imagen, (x0, y0 + radio), (x0, y1 - radio), color, grosor, cv2.LINE_AA)
        cv2.line(imagen, (x1, y0 + radio), (x1, y1 - radio), color, grosor, cv2.LINE_AA)
    for centro, angulo in (((x0 + radio, y0 + radio), 180), ((x1 - radio, y0 + radio), 270),
                           ((x1 - radio, y1 - radio), 0), ((x0 + radio, y1 - radio), 90)):
        cv2.ellipse(imagen, centro, (radio, radio), angulo, 0, 90, color, grosor, cv2.LINE_AA)
    return imagen


def caja_esquinas(imagen, caja, color=VERDE, grosor=3, largo=0.22):
    """Marco tipo visor: solo las cuatro esquinas. Tapa menos cara que un
    rectangulo entero, que importa cuando encima va un mapa de calor."""
    x, y, ancho, alto = [int(v) for v in caja]
    paso = int(max(8, min(ancho, alto) * largo))
    for (px, py), (dx, dy) in (
        ((x, y), (1, 1)), ((x + ancho, y), (-1, 1)),
        ((x, y + alto), (1, -1)), ((x + ancho, y + alto), (-1, -1)),
    ):
        cv2.line(imagen, (px, py), (px + dx * paso, py), color, grosor, cv2.LINE_AA)
        cv2.line(imagen, (px, py), (px, py + dy * paso), color, grosor, cv2.LINE_AA)
    return imagen


def barra(imagen, x, y, ancho, alto, fraccion, color=ACENTO, fondo=(48, 44, 40)):
    """Barra horizontal de progreso, para el ranking de regiones."""
    rect_redondeado(imagen, (x, y), (x + ancho, y + alto), fondo, radio=alto // 2)
    lleno = int(ancho * max(0.0, min(1.0, fraccion)))
    if lleno > alto // 2:
        rect_redondeado(imagen, (x, y), (x + lleno, y + alto), color, radio=alto // 2)
    return imagen


def panel_traslucido(frame, p0, p1, color=FONDO_BARRA, alpha=0.72, radio=12):
    """Rectangulo semitransparente sobre el video (para cajas de texto)."""
    x0, y0 = max(0, int(p0[0])), max(0, int(p0[1]))
    x1 = min(frame.shape[1], int(p1[0]))
    y1 = min(frame.shape[0], int(p1[1]))
    if x1 <= x0 or y1 <= y0:
        return frame
    region = frame[y0:y1, x0:x1]
    capa = np.empty_like(region)
    capa[:] = color
    if radio > 0:
        mascara = np.zeros(region.shape[:2], np.uint8)
        rect_redondeado(mascara, (0, 0), (x1 - x0 - 1, y1 - y0 - 1), 255, radio=radio)
        mascara = (mascara.astype(np.float32) / 255.0 * alpha)[..., None]
    else:
        mascara = alpha
    frame[y0:y1, x0:x1] = np.clip(region * (1 - mascara) + capa * mascara, 0, 255).astype(np.uint8)
    return frame
