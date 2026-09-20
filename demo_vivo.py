"""
Demo en vivo de estimacion de edad — Semana de la Salud Mental (UdeSA).

Toma la webcam, detecta caras, y muestra la edad estimada en tiempo real.
Pensado para un stand: robusto a luz/angulos (usa YuNet si esta disponible),
con una aclaracion en pantalla sobre los limites del sistema.

Uso tipico (con la webcam):
    python demo_vivo.py

Probar sobre una imagen (sin webcam):
    python demo_vivo.py --image fotosprueba/1_0_0_20161219204512212.jpg.chip.jpg

Bajar el error con ensemble de los 2 mejores modelos:
    python demo_vivo.py --ensemble

Teclas en la ventana de webcam:  q = salir
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

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

# Modelo YuNet (opcional). Si el archivo no esta, se cae a Haar cascade.
YUNET_PATH = os.path.join('models', 'face_detection_yunet_2023mar.onnx')

_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


# ----------------------------------------------------------------------------
# Deteccion de caras
# ----------------------------------------------------------------------------
class FaceDetector:
    """Detector de caras. Usa YuNet (robusto) si el modelo onnx esta presente;
    si no, cae a Haar cascade (peor, pero viene con OpenCV)."""

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
            self.mode = 'haar'
        print(f"[i] Detector de caras: {self.mode}")

    def detect(self, frame_bgr):
        """Devuelve lista de cajas (x, y, w, h) en pixeles."""
        if self.mode == 'yunet':
            h, w = frame_bgr.shape[:2]
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(frame_bgr)
            if faces is None:
                return []
            return [tuple(map(int, f[:4])) for f in faces]
        # Haar
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=8, minSize=(80, 80)
        )
        return [tuple(map(int, f)) for f in faces]


def crop_face(frame_bgr, box, margin=0.2):
    """Recorta la cara con un margen y la deja cuadrada, para que el encuadre
    se parezca al del dataset de entrenamiento (UTKFace)."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * margin)
    half = side / 2
    x0 = int(max(0, cx - half))
    y0 = int(max(0, cy - half))
    x1 = int(min(frame_bgr.shape[1], cx + half))
    y1 = int(min(frame_bgr.shape[0], cy + half))
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop


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
def predict_age(models, crop_bgr, tta=True):
    """Predice edad (float) promediando modelos y, si tta, la imagen espejada."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    tensors = [_transform(pil)]
    if tta:
        tensors.append(_transform(pil.transpose(Image.FLIP_LEFT_RIGHT)))
    batch = torch.stack(tensors).to(DEVICE)

    preds = []
    for model in models:
        out = model(batch)             # [N, 1]
        preds.append(out.mean().item())  # promedia original + flip
    return float(np.mean(preds))


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
# Dibujo
# ----------------------------------------------------------------------------
def draw(frame, box, age):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
    label = f"~{age} anios"
    bar_w = max(w, 150)
    # Si no hay lugar arriba de la caja, dibujo la etiqueta adentro (arriba).
    if y - 34 < 0:
        y0, y1, ty = y, y + 34, y + 25
    else:
        y0, y1, ty = y - 34, y, y - 9
    cv2.rectangle(frame, (x, y0), (x + bar_w, y1), (0, 200, 0), -1)
    cv2.putText(frame, label, (x + 6, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


def draw_disclaimer(frame):
    txt = "Estimacion aproximada. Puede equivocarse mucho segun luz, angulo, edad y rasgos."
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 30), (frame.shape[1], h), (30, 30, 30), -1)
    cv2.putText(frame, txt, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Modos
# ----------------------------------------------------------------------------
def run_image(models, detector, image_path, margin):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[x] No pude abrir la imagen: {image_path}")
        return
    boxes = detector.detect(frame)
    if not boxes:
        # Sin cara detectada: probamos sobre la imagen entera (util para chips UTKFace).
        print("[i] No detecte cara; predigo sobre la imagen completa.")
        age = clamp_age(predict_age(models, frame))
        print(f"    Edad estimada: {age} anios")
        return
    for box in boxes:
        crop = crop_face(frame, box, margin)
        if crop is None:
            continue
        age = clamp_age(predict_age(models, crop))
        print(f"    Cara en {box} -> {age} anios")
        draw(frame, box, age)
    out = 'demo_output.jpg'
    cv2.imwrite(out, frame)
    print(f"[ok] Guarde el resultado en {out}")


def run_webcam(models, detector, camera, margin, smooth=True):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"[x] No pude abrir la camara {camera}.")
        return
    print("[i] Webcam abierta. Teclas: 'q' salir | 's' prender/apagar suavizado.")
    smoother = AgeSmoother()
    last = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # espejo, mas natural

        # 1) Detectar + predecir (crudo) por cada cara.
        dets = []
        for box in detector.detect(frame):
            crop = crop_face(frame, box, margin)
            if crop is None:
                continue
            dets.append((box, predict_age(models, crop)))

        # 2) Suavizar en el tiempo (o no) y dibujar.
        if smooth:
            shown = smoother.update(dets)
        else:
            shown = [(box, clamp_age(raw)) for box, raw in dets]
        for box, age in shown:
            draw(frame, box, age)

        draw_disclaimer(frame)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        estado = "ON" if smooth else "OFF (crudo)"
        cv2.putText(frame, f"{fps:4.1f} fps  [{DEVICE}]  suavizado: {estado}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Estimacion de edad — Semana de la Salud Mental", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            smooth = not smooth
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
    ap.add_argument('--no-smooth', dest='smooth', action='store_false',
                    help="Arrancar sin suavizado temporal (muestra la prediccion cruda, inestable).")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"[x] No encuentro los pesos: {args.model}")
        print("    Baja los pesos del Drive (ver codigo/README.md) y ponelos en checkpoints/.")
        sys.exit(1)

    print(f"[i] Device: {DEVICE}")
    models = [load_arista(args.model)]
    if args.ensemble:
        if os.path.exists(args.model2):
            models.append(load_finetuneada(args.model2))
            print("[i] Ensemble activo: arista + finetuneada")
        else:
            print(f"[!] No encontre {args.model2}; sigo solo con arista.")

    detector = FaceDetector(prefer=args.detector)

    if args.image:
        run_image(models, detector, args.image, args.margin)
    else:
        run_webcam(models, detector, args.camera, args.margin, smooth=args.smooth)


if __name__ == '__main__':
    main()
