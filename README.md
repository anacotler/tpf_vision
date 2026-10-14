# Estimación de edad con explicabilidad

Estimación de edad a partir de la cara, entrenada sobre UTKFace, con una demo
en vivo (webcam) que muestra la edad estimada y **qué parte de la cara mira la
red** (Grad-CAM u oclusión).

## Estructura

```
├── README.md
├── pyproject.toml         paquete instalable (src/age_estimation)
├── requirements.txt
├── data/
│   ├── raw/               imágenes de UTKFace (no va al repo)
│   ├── processed/         split train/valid/test y lista completa (CSV)
│   └── samples/           algunas caras de UTKFace para probar rápido
├── models/                pesos entrenados + detector YuNet (no va al repo)
├── notebooks/             análisis exploratorio, Integrated Gradients, pruebas
├── reports/               capturas de la demo, historias de entrenamiento (no va al repo)
├── archive/               scripts viejos que ya no se mantienen
└── src/age_estimation/
    ├── config.py          configuración de entrenamiento
    ├── dataset.py         dataset y dataloaders de UTKFace
    ├── explainability.py  Grad-CAM, oclusión, atribución por región
    ├── models/            arquitecturas: arista, resnet101, frozen, finetuned, self_attention
    ├── training/          train.py, train_resnet101.py, engine.py, hyperparameter_tuning.py
    └── demo/              demo en vivo: app.py (pipeline), ui.py (dibujo)
```

## Instalación

```bash
pip install -e .           # solo la demo
pip install -r requirements.txt   # demo + entrenamiento (-e .[train])
```

### Pesos

Van en `models/`:

| Archivo | Qué es |
|---|---|
| `arista.pt` | ResNet-50 a 128 px (modelo original) |
| `resnet101_robusta.pt` | ResNet-101 a 200 px con aumentación tipo webcam (la que usa la demo) |
| `resnet101.pt` | ResNet-101 a 200 px sin esa aumentación |
| `face_detection_yunet_2023mar.onnx` | detector de caras YuNet |

YuNet se baja de OpenCV Zoo:

```bash
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

### Datos (solo para entrenar o evaluar)

UTKFace, versión "Aligned & Cropped" (Kaggle `jangedoo/utkface-new`), con las
imágenes descomprimidas en `data/raw/UTKFace/`.

## Demo en vivo

```bash
python -m age_estimation.demo                    # webcam
python -m age_estimation.demo --image foto.jpg   # una imagen; guarda reports/captures/demo_output.jpg
python -m age_estimation.demo --help             # todas las opciones
```

Teclas: `q` salir, `espacio` congelar, `e` explicación on/off, `m` Grad-CAM ↔
oclusión, `l` capa del Grad-CAM, `o` mapa también sobre el video, `s`
suavizado, `g` guardar captura (pantalla + frame crudo), `h` ayuda.

Qué hace por cada frame:

1. Detecta caras con YuNet sobre el frame achicado a 640 px e ignora las de
   menos de 120 px de ancho (`--cara-min`).
2. Recorta la cara alineada con los 5 landmarks, reproduciendo el encuadre de
   UTKFace. Los landmarks se suavizan en el tiempo para que el recorte no
   tiemble.
3. Adapta la imagen de la webcam a cómo son las fotos de UTKFace: la achica a
   64 px (`--reducir`) y le estira el contraste (`--sin-autocontraste` para
   apagarlo). Sin esto la red subestima mucho la edad: lee el detalle y la
   imagen "chata" de la webcam como piel joven.
4. Predice con el promedio de arista y la ResNet-101 robusta
   (`--modelo combinado|arista|resnet101`) y suaviza la edad en el tiempo
   (`--suavizado ema|mediana`).

## Entrenamiento

Se corre desde la raíz del repo:

```bash
# la ResNet-101 de la demo (~30 min en una RTX 3060)
python -m age_estimation.training.train_resnet101 --aug fuerte --epocas 30 \
    --nombre resnet101_robusta --salida models/resnet101_robusta.pt
```

Todos los modelos usan el split de `data/processed/`.

## Resultados (test de `data/processed/`, 4654 caras)

| Modelo | MAE (años) |
|---|---|
| arista (ResNet-50, 128 px) | 5.11 |
| ResNet-101, 200 px | 4.95 |
| ResNet-101 robusta, 200 px | 4.91 |
| **Combinado: arista + ResNet-101 robusta** | **4.78** |

Medido sobre las fotos de UTKFace tal cual. Con el preprocesamiento de la demo
para webcam (paso 3) el MAE sobre UTKFace sube a ~5.3.

> **Ojo con `archive/test.csv`**: es otro split y ~2400 de sus caras están en el
> train de arista, así que medir ahí da un MAE optimista (4.13).
