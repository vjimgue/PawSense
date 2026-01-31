import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Configuración de la página
st.set_page_config(
    page_title="PawSense AI - Identificador de Razas",
    page_icon="🐾",
    layout="centered"
)

st.title("🐾 PawSense AI")
st.write("Sube una foto de un perro y te diré su raza.")

# Rutas de archivos
MODEL_PATH = "model/modelo_perros.keras"
CLASS_NAMES_PATH = "model/class_names.json"

@st.cache_resource
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        return None, None
    
    # Cargar modelo y lista de clases
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model_and_classes()

if model is None:
    st.error("⚠️ No se encontró el modelo o los nombres de clases.")
    st.info("Por favor, ejecuta el notebook `modelo_prediccion_perros.ipynb` (Fases 5 y 6) para entrenar y guardar el modelo.")
    st.code("model.save('modelo_perros.keras')\n# y guardar class_names.json", language="python")
else:
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Mostrar imagen
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imagen subida", use_column_width=True)
            
            # Preprocesamiento
            st.write("🔍 Analizando...")
            # El modelo espera (224, 224, 3) y la capa de preprocesamiento normaliza
            img_array = image.resize((224, 224))
            img_array = tf.keras.utils.img_to_array(img_array)
            img_array = tf.expand_dims(img_array, 0) # Crear un batch

            # Predicción
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            
            predicted_class = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)

            st.success(f"¡Es un **{predicted_class}**!")
            st.write(f"Confianza del modelo: **{confidence:.2f}%**")
            
            # Mostrar top 3 probabilidades
            with st.expander("Ver otras posibilidades"):
                top_3_indices = np.argsort(score)[-3:][::-1]
                for i in top_3_indices:
                    st.write(f"- {class_names[i]}: {100 * score[i]:.2f}%")
                    
        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")