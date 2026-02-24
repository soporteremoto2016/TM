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
