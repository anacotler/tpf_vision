
"""
import os

def detect_and_crop_faces(image_path, output_dir):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Read the input image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: No se pudo cargar la imagen. Verifica la ruta.")
        return
    
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Filtrar detecciones basadas en el tamaño del bounding box
    min_width, min_height = 200, 200  # Tamaño mínimo para considerar como cara
    faces = [f for f in faces if f[2] >= min_width and f[3] >= min_height]

    
    if len(faces) == 0:
        print("No se detectaron caras en la imagen.")
        return
    
    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        roi_color = img[y:y+h, x:x+w]
        # Save the cropped face
        output_path = os.path.join(output_dir, f"face_{i+1}.jpg")
        cv2.imwrite(output_path, roi_color)
        print(f"Rostro guardado en: {output_path}")
    
    # Optionally, display the image with rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('Detected Faces', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = '/home/anacotler/Documents/vision_artificial/tpf_vision/sere.jpg'
output_dir = '/home/anacotler/Documents/vision_artificial/tpf_vision/cropped_faces'
os.makedirs(output_dir, exist_ok=True)
detect_and_crop_faces(image_path, output_dir)
"""

import cv2
import os
import numpy as np

def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    return enhanced_gray


def increase_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = cv2.min(v, 255)
    final_hsv = cv2.merge((h, s, v))
    bright_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bright_img, cv2.COLOR_BGR2GRAY)
    return gray


def detect_faces_and_eyes_opencv(image_path, output_dir):
    """
    Detecta caras y ojos en una imagen usando OpenCV, guarda las caras recortadas y muestra las detecciones.
    Args:
        image_path (str): Ruta de la imagen de entrada.
        output_dir (str): Ruta del directorio donde se guardarán las caras recortadas.
    """
    # Cargar los clasificadores Haar para cara y ojos
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Leer la imagen
    img = cv2.imread(image_path)
    if img is None:
        print("Error: No se pudo cargar la imagen. Verifica la ruta.")
        return
    
    # Aplicar filtro Gaussiano para reducir ruido
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    avg_brightness = np.mean(img_blur)

    if avg_brightness < 50:
        gray = increase_brightness(img_blur, value=50)

    elif avg_brightness < 100:
        gray = enhance_contrast(img_blur)
    
    else:
        gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        

    # Detectar caras
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10, minSize=(80, 80))

    if len(faces) == 0:
        print("No se detectaron caras en la imagen.")
        return
    
    print(f"Se detectaron {len(faces)} cara(s).")
    
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Procesar cada cara detectada
    for i, (x, y, w, h) in enumerate(faces):
        # Recortar la cara
        cropped_face = img[y:y+h, x:x+w]
        
        # Guardar la cara recortada
        output_path = os.path.join(output_dir, f"face_{i+1}.jpg")

        
        # Detectar ojos en la región de la cara
        face_gray = gray[y:y+h, x:x+w]
        face_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))

        #revisar si se detectaron ojos, si no se detectan la imagen no es una face
        if len(eyes) != 0:
            cv2.imwrite(output_path, cropped_face)
            print(f"Rostro {i+1} guardado en: {output_path}")
                
            
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    # Dibujar rectángulos alrededor de las caras y ojos en la imagen original
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Cara
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)  # Ojos
    
    # Mostrar la imagen con las detecciones
    cv2.imshow('Detected Faces and Eyes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Rutas de ejemplo
image_path = '/home/anacotler/Documents/vision_artificial/tpf_vision/pibes.jpg'
output_dir = '/home/anacotler/Documents/vision_artificial/tpf_vision/cropped_faces'
os.makedirs(output_dir, exist_ok=True)

# Llamar a la función
detect_faces_and_eyes_opencv(image_path, output_dir)
