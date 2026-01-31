import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import tensorflow as tf
import numpy as np
import streamlit.components.v1 as components

# --- Configuration ---
MODEL_PATH = "model/modelo_perros_pytorch.pth"
TF_MODEL_PATH = "model/modelo_prediccion_perros_v1.keras"
CLASS_NAMES_PATH = "model/class_names.json"
TRANSLATIONS_PATH = "model/breed_translations.json"

st.set_page_config(page_title="PawSense", page_icon="🐶", initial_sidebar_state="collapsed")

# --- Utils ---
@st.cache_resource
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        st.error(f"File not found: {CLASS_NAMES_PATH}")
        return []
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return class_names

@st.cache_resource
def load_translations():
    if not os.path.exists(TRANSLATIONS_PATH):
        return {}
    with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

import base64

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_img_as_base64(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache_resource
def load_model(num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Rebuild the model structure (EfficientNet B0)
    # We use weights=None because we will load our own state_dict
    # Note: efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1) was used in training initialization
    # but since we are loading a state_dict, we can start with empty weights or default.
    # To be safe and match the architecture exactly as `models.efficientnet_b0` provides:
    model = models.efficientnet_b0(weights=None)
    
    # Modify the classifier head to match the training script
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None, device

    # Load state dict
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

    model.to(device)
    model.eval()
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def load_tf_model():
    if not os.path.exists(TF_MODEL_PATH):
        st.error(f"TF Model file not found: {TF_MODEL_PATH}")
        return None
    try:
        model = tf.keras.models.load_model(TF_MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading TF model: {e}")
        return None

def process_image(image):
    # Standard transformers for EfficientNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

def process_image_tf(image):
    # Resize to 224x224 matches the PyTorch flow, assuming TF model was trained similarly
    # If using MobileNet or EfficientNet in TF, usually expects 224x224
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create batch axis
    return img_array

# --- Main UI ---
st.title("🐶 PawSense")
set_background("assets/background.png")
local_css("assets/style.css")
st.write("Sube tu imagen de un perro y descubre su raza con nuestro clasificador.")

class_names = load_class_names()
translations = load_translations()

# Sidebar Config
st.sidebar.header("⚙️ Configuración")
popup_duration = st.sidebar.slider("Tiempo del Popup (s)", 3, 10, 5)

if class_names:
    model, device = load_model(len(class_names))
    tf_model = load_tf_model()

    if model and tf_model:
        uploaded_file = st.file_uploader("Elige una imagen de un perro...", type=["jpg", "jpeg", "png"])
        
        # Check if the user wants to use a default image for testing (optional, not requested but good for quick check)
        # But per plan, just file uploader.

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Imagen cargada', width="stretch")
            
            
            with st.spinner('Analizando imagen...'):
                # --- Inference PyTorch ---
                input_tensor = process_image(image).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                top3_prob, top3_idx = torch.topk(probabilities, 3)
                
                pytorch_winner_idx = top3_idx[0].item()
                pytorch_winner_prob = top3_prob[0].item()
                pytorch_winner_key = class_names[pytorch_winner_idx]
                
                # --- Inference TensorFlow ---
                input_tensor_tf = process_image_tf(image)
                predictions_tf = tf_model.predict(input_tensor_tf)
                probabilities_tf = predictions_tf[0]
                
                top3_idx_tf = probabilities_tf.argsort()[-3:][::-1]
                top3_prob_tf = probabilities_tf[top3_idx_tf]
                
                tf_winner_idx = top3_idx_tf[0]
                tf_winner_prob = float(top3_prob_tf[0])
                tf_winner_key = class_names[tf_winner_idx]

                # --- Determine Grand Winner ---
                if pytorch_winner_prob > tf_winner_prob:
                    grand_winner_key = pytorch_winner_key
                    grand_winner_prob = pytorch_winner_prob
                    grand_winner_source = "PyTorch"
                else:
                    grand_winner_key = tf_winner_key
                    grand_winner_prob = tf_winner_prob
                    grand_winner_source = "TensorFlow"

                grand_winner_name = translations.get(grand_winner_key, grand_winner_key.replace("_", " "))
                
                st.balloons()
                
                
                # Custom Centered Popup with Close Button and Timer
                # Using a unique ID 'grand-winner-popup' to target with JS
                # Note: We remove inline onclick as it might be stripped. We use a separate component to inject logic.
                # Get names for popup
                pytorch_winner_breed = translations.get(pytorch_winner_key, pytorch_winner_key.replace("_", " "))
                tf_winner_breed = translations.get(tf_winner_key, tf_winner_key.replace("_", " "))

                # Custom Centered Popup with Close Button and Timer
                # Using a unique ID 'grand-winner-popup' to target with JS
                # Note: We remove inline onclick as it might be stripped. We use a separate component to inject logic.
                popup_html = f"""
<div id="grand-winner-popup" class="grand-winner-popup">
<span id="popup-close-btn" class="popup-close-btn">&times;</span>
<h1 style="color: #e67e22; margin-bottom: 10px;">¡TENEMOS UN GANADOR! </h1>
<h2 style="color: #2c3e50; font-size: 2.5rem; margin: 10px 0;">{grand_winner_name}</h2>
<p style="color: #7f8c8d; font-size: 1.2rem;">Confianza Global: <strong>{grand_winner_prob*100:.2f}%</strong></p>
<p style="color: #95a5a6; font-size: 0.9rem; margin-bottom: 20px;">(Mejor coincidencia: {grand_winner_source})</p>
<div class="popup-columns">
<div class="popup-col">
<h3>PyTorch</h3>
<p><strong>{pytorch_winner_breed}</strong></p>
<p>{pytorch_winner_prob*100:.2f}%</p>
</div>
<div class="popup-col">
<h3>TensorFlow</h3>
<p><strong>{tf_winner_breed}</strong></p>
<p>{tf_winner_prob*100:.2f}%</p>
</div>
</div>
<div class="popup-progress-container">
<div id="popup-progress-bar" class="popup-progress-bar"></div>
</div>
</div>
"""
                st.markdown(popup_html, unsafe_allow_html=True)
                
                # Inject JS via iframe to ensure execution and access to parent DOM
                js_code = f"""
                <script>
                    const duration = {popup_duration};
                    const popupId = 'grand-winner-popup';
                    const btnId = 'popup-close-btn';
                    const progressId = 'popup-progress-bar';
                    
                    // Access parent document
                    const doc = window.parent.document;
                    
                    const popup = doc.getElementById(popupId);
                    const btn = doc.getElementById(btnId);
                    const progressBar = doc.getElementById(progressId);
                    
                    // Close button handler
                    if (btn) {{
                        btn.onclick = function() {{
                            if (popup) popup.style.display = 'none';
                        }};
                    }}
                    
                    // Animation logic
                    if (popup && progressBar) {{
                        // Set transition duration to match popup duration
                        progressBar.style.transition = `width ${{duration}}s linear`;
                        
                        // Force reflow to ensure transition works
                        void progressBar.offsetWidth; 
                        
                        // Start animation
                        setTimeout(() => {{
                            progressBar.style.width = '0%';
                        }}, 100);
                        
                        // Fade out trigger AFTER progress bar finishes
                        setTimeout(() => {{
                            popup.classList.add('fade-out');
                        }}, duration * 1000);
                        
                        // Remove element after fade out (duration + 0.5s for existing CSS transition)
                        setTimeout(() => {{
                            popup.style.display = 'none';
                        }}, (duration + 0.5) * 1000);
                    }}
                </script>
                """
                components.html(js_code, height=0, width=0)

                col1, col2 = st.columns(2)
                
                # --- Display PyTorch ---
                with col1:
                    st.header("PyTorch")
                    winner_name = translations.get(pytorch_winner_key, pytorch_winner_key.replace("_", " "))
                    
                    st.subheader("Resultados:")
                    for i in range(3):
                        prob = top3_prob[i].item()
                        idx = top3_idx[i].item()
                        breed_key = class_names[idx]
                        breed_name = translations.get(breed_key, breed_key.replace("_", " "))
                        
                        card_class = "prediction-card winner-card" if i == 0 else "prediction-card"
                        emoji = "🥇 " if i == 0 else ""
                        
                        st.markdown(f"""
                        <div class="{card_class}">
                            <span class="breed-name">{emoji}{i+1}. {breed_name}</span>
                            <br>
                            <small>Confianza: {prob*100:.2f}%</small>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(prob)

                # --- Display TensorFlow ---
                with col2:
                    st.header("TensorFlow")
                    winner_name_tf = translations.get(tf_winner_key, tf_winner_key.replace("_", " "))

                    st.subheader("Resultados:")
                    for i in range(3):
                        prob = float(top3_prob_tf[i])
                        idx = top3_idx_tf[i]
                        breed_key = class_names[idx]
                        breed_name = translations.get(breed_key, breed_key.replace("_", " "))
                        
                        card_class = "prediction-card winner-card" if i == 0 else "prediction-card"
                        emoji = "🥇 " if i == 0 else ""
                        
                        st.markdown(f"""
                        <div class="{card_class}">
                            <span class="breed-name">{emoji}{i+1}. {breed_name}</span>
                            <br>
                            <small>Confianza: {prob*100:.2f}%</small>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(prob)

    else:
        st.warning("Model could not be loaded. Please check the file paths.")
else:
    st.warning("Class names could not be loaded.")
