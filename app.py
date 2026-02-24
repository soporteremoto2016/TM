import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

# 1. CONFIGURACIÓN DE PÁGINA (Debe ser lo primero)
st.set_page_config(
    page_title="AI Vehicle Classifier",
    page_icon="🚗",
    layout="centered"
)

# 2. CSS PERSONALIZADO PARA DISEÑO ELEGANTE (Fondo Azul y Estilo)
st.markdown("""
    <style>
    /* Fondo principal en degradado azul */
    .stApp {
        background: linear-gradient(to bottom, #001f3f, #0074D9);
        color: white;
    }
    
    /* Estilo para el título */
    .main-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #FFFFFF;
        text-align: center;
        font-weight: bold;
        padding: 20px;
        text-shadow: 2px 2px 4px #000000;
    }
    
    /* Contenedor de resultados */
    .result-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_stdio=True, unsafe_allow_html=True)

# 3. CARGA DE MODELO
@st.cache_resource # Esto evita que el modelo se recargue cada vez que mueves algo
def load_my_model():
    return load_model('keras_model.h5')

model = load_my_model()

# 4. ENCABEZADO ORGANIZADO
st.markdown('<h1 class="main-title">🚀 Reconocimiento Inteligente de Vehículos</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Sube una imagen y nuestra IA identificará si es un <b>Auto, Moto o Bicicleta</b></p>", unsafe_allow_html=True)

# 5. BARRA LATERAL (SIDEBAR)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2345/2345454.png", width=100) # Icono decorativo
    st.title("Panel de Control")
    st.info("Este modelo utiliza una red neuronal convolucional entrenada en Teachable Machine.")
    st.divider()
    st.write("💻 **Versión:** 2.0 (Diseño Elegante)")

# 6. CARGA DE ARCHIVO
st.divider()
img_file_buffer = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

if img_file_buffer is not None:
    # Columnas para organizar imagen vs resultados
    col_img, col_res = st.columns([1, 1])

    with col_img:
        img = Image.open(img_file_buffer).convert("RGB")
        size = (224, 224)
        img_display = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
        st.image(img_display, caption="Imagen Detectada", use_container_width=True)

    # PROCESAMIENTO
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img_array = np.array(img_display)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    
    prediction = model.predict(data)

    with col_res:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Análisis de la IA")
        
        # Lógica de las tres condiciones
        if prediction[0][0] > 0.5:
            st.success(f"🚗 **AUTO**\n\nConfianza: {prediction[0][0]:.2%}")
        elif prediction[0][1] > 0.5:
            st.success(f"🏍️ **MOTO**\n\nConfianza: {prediction[0][1]:.2%}")
        elif prediction[0][2] > 0.5:
            st.success(f"🚲 **BICICLETA**\n\nConfianza: {prediction[0][2]:.2%}")
        else:
            st.warning("⚠️ No se pudo determinar con precisión.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # 7. MÉTRICAS DETALLADAS (Abajo)
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Prob. Auto", f"{prediction[0][0]:.1%}")
    m2.metric("Prob. Moto", f"{prediction[0][1]:.1%}")
    m3.metric("Prob. Bici", f"{prediction[0][2]:.1%}")
