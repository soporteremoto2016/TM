import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps 
from keras.models import load_model
import platform

# Mostrar versión de Python
st.write("Versión de Python:", platform.python_version())

# Cargar el modelo
model = load_model('keras_model.h5')

st.title("Reconocimiento de Imágenes")

# Imagen de portada opcional
try:
    image_portada = Image.open('OIG5.jpg')
    st.image(image_portada, width=350)
except:
    st.warning("No se encontró la imagen de portada OIG5.jpg")

with st.sidebar:
    st.subheader("Sube una imagen para que el modelo de Teachable Machine la identifique.")

# --- CAMBIO CLAVE AQUÍ ---
# Se cambia st.camera_input por st.file_uploader
img_file_buffer = st.file_uploader("Carga una imagen", type=['jpg', 'jpeg', 'png'])

if img_file_buffer is not None:
    # 1. Preparar el array para el modelo (Teachable Machine usa 224x224x3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # 2. Leer el archivo como una imagen PIL
    img = Image.open(img_file_buffer).convert("RGB") # Convertimos a RGB por si suben un PNG con transparencia

    # 3. Redimensionar usando ImageOps para mantener la proporción (recomendado)
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    # Mostrar la imagen que se acaba de subir
    st.image(img, caption="Imagen para procesar", width=300)

    # 4. Convertir a array de numpy
    img_array = np.array(img)

    # 5. Normalizar la imagen (basado en el estándar de Teachable Machine)
    # Algunos modelos usan / 127.5 - 1, otros / 255. Verifica tu modelo original.
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    
    # 6. Cargar en el array de datos
    data[0] = normalized_image_array

    # 7. Ejecutar la inferencia
    prediction = model.predict(data)
    
    # 8. Mostrar resultados
    st.divider()
    if prediction[0][0] > 0.5:
        st.header(f'Izquierda (Probabilidad: {prediction[0][0]:.2f})')
    if prediction[0][1] > 0.5:
        st.header(f'Arriba (Probabilidad: {prediction[0][1]:.2f})')
