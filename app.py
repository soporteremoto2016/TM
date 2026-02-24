import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import platform

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Sistema de Clasificación de Imagenes",
    page_icon="🚗",
    layout="centered"
)

# 2. CSS PERSONALIZADO 
st.markdown("""
    <style>
    /* 1. Fondo principal y texto general */
    .stApp {
        background-color: #001f3f;
        background-image: linear-gradient(180deg, #001f3f 0%, #0074D9 100%);
    }
    
    .stApp p, .stApp h1, .stApp h2, .stApp h3 {
        color: white !important;
    }

    /* 2. CAMBIO ESPECÍFICO: Texto del Drag and Drop a AZUL */
    /* Cambia el texto "Drag and drop file here" */
    .stFileUploader section [data-testid="stWidgetLabel"] p {
        color: #0074D9 !important; /* Un azul vibrante */
        font-weight: bold;
    }
    
    /* Cambia el texto pequeño (límite de 200MB, etc) */
    .stFileUploader section div div {
        color: #001f3f !important; /* Azul oscuro para que resalte en el fondo gris claro del widget */
    }

    /* 3. Estilo del recuadro del cargador */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(255, 255, 255, 0.9); /* Fondo casi blanco para que el texto azul se vea */
        border: 2px dashed #0074D9;
        border-radius: 10px;
    }

    /* Título elegante */
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. CARGA DE MODELO (Con cache para evitar lentitud)
@st.cache_resource
def load_my_model():
    return load_model('keras_model.h5')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# 4. ENCABEZADO
st.markdown('<h1 class="main-title">🚗 AI Vehicle Classifier</h1>', unsafe_allow_html=True)
st.write("---")

# 5. CARGA DE ARCHIVO
img_file_buffer = st.file_uploader("Sube una imagen (Auto, Moto o Bicicleta)", type=['jpg', 'jpeg', 'png'])

if img_file_buffer is not None:
    # Columnas para organizar imagen y resultado
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        # Procesamiento de imagen
        image = Image.open(img_file_buffer).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        st.image(image, caption="Imagen cargada", use_container_width=True)

    with col2:
        # Preparación para el modelo
        img_array = np.asarray(image)
        normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Inferencia
        with st.spinner('Analizando imagen...'):
            prediction = model.predict(data)
        
        st.subheader("Resultados del Análisis")
        
        # Lógica de detección (Asegúrate que el orden [0,1,2] coincida con tu modelo)
        if prediction[0][0] > 0.5:
            st.success(f"### Es un AUTO 🚗")
            st.write(f"Confianza: **{prediction[0][0]:.2%}**")
        elif prediction[0][1] > 0.5:
            st.success(f"### Es una MOTO 🏍️")
            st.write(f"Confianza: **{prediction[0][1]:.2%}**")
        elif prediction[0][2] > 0.5:
            st.success(f"### Es una BICICLETA 🚲")
            st.write(f"Confianza: **{prediction[0][2]:.2%}**")
        else:
            st.warning("No estoy seguro. La confianza es muy baja.")

    # Métricas adicionales al final
    st.write("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Auto", f"{prediction[0][0]:.1%}")
    m2.metric("Moto", f"{prediction[0][1]:.1%}")
    m3.metric("Bici", f"{prediction[0][2]:.1%}")
