import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

# 1. Cargar el modelo
# Asegúrate de que este modelo fue entrenado con las 3 clases
model = load_model('keras_model.h5')

st.title("Clasificador: Autos, Motos y Bicicletas")

# 2. Carga de archivo
img_file_buffer = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png'])

if img_file_buffer is not None:
    # Preparar el contenedor para la imagen (Teachable Machine usa 224x224x3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Leer y procesar la imagen
    img = Image.open(img_file_buffer).convert("RGB")
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    st.image(img, caption="Imagen procesada", width=300)

    # Convertir a array y normalizar
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # 3. Ejecutar la inferencia (Predicción)
    prediction = model.predict(data)
    
    st.divider()

    # 4. Lógica de las TRES condiciones
    # Nota: El orden [0], [1], [2] depende de cómo entrenaste las etiquetas en Teachable Machine
    
    col1, col2, col3 = st.columns(3) # Creamos columnas para ver los porcentajes

    with col1:
        st.metric("Auto", f"{prediction[0][0]:.2%}")
    with col2:
        st.metric("Moto", f"{prediction[0][1]:.2%}")
    with col3:
        st.metric("Bicicleta", f"{prediction[0][2]:.2%}")

    # Mostrar el resultado principal
    if prediction[0][0] > 0.5:
        st.success(f"🚗 Resultado: **Auto** (Confianza: {prediction[0][0]:.2f})")
        
    elif prediction[0][1] > 0.5:
        st.success(f"🏍️ Resultado: **Moto** (Confianza: {prediction[0][1]:.2f})")
        
    elif prediction[0][2] > 0.5:
        st.success(f"🚲 Resultado: **Bicicleta** (Confianza: {prediction[0][2]:.2f})")
        
    else:
        st.warning("🤔 No se pudo identificar claramente. La confianza es muy baja.")
