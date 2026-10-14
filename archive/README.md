# archive/

Scripts y archivos viejos del proyecto, guardados tal cual. **No se mantienen
y la mayoria no corre**: tienen rutas absolutas de otra maquina
(`/home/anacotler/...`) e importan modulos que se movieron a `src/age_estimation/`.

- `IG_*.py`, `gradcam.py`: explicabilidad (Integrated Gradients, Grad-CAM) de
  la primera version. La de la demo esta en
  `src/age_estimation/explainability.py`.
- `inference.py`, `predecir_una_imagen.py`, `detect_and_crop_face.py`:
  inferencia sobre imagenes sueltas. Para eso hoy se usa
  `python -m age_estimation.demo --image foto.jpg`.
- `model.py`, `model2.py`, `model3.py`: duplicados de `models/frozen.py`,
  `models/arista.py` (cambia solo el nombre de la clase) y
  `models/self_attention.py` (en `src/age_estimation/`).
- `split_data.py`, `train.csv`, `validation.csv`, `test.csv`: un split
  distinto al de `data/processed/`. **Ojo**: ~2400 caras de este `test.csv`
  estan en el train con el que se entreno arista, asi que medir ahi da un MAE
  optimista (4.13 en vez de 5.11).
