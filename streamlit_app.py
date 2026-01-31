import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# --- Configuration ---
MODEL_PATH = "model/modelo_perros.pth"
CLASS_NAMES_PATH = "model/class_names.json"
TRANSLATIONS_PATH = "model/breed_translations.json"

st.set_page_config(page_title="PawSenses", page_icon="🐶")

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

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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
    return model, device

def process_image(image):
    # Standard transformers for EfficientNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

# --- Main UI ---
st.title("🐶 PawSense")
local_css("assets/style.css")
st.write("Sube tu imagen de un perro y descubre su raza con nuestro clasificador.")

class_names = load_class_names()
translations = load_translations()

if class_names:
    model, device = load_model(len(class_names))

    if model:
        uploaded_file = st.file_uploader("Elige una imagen de un perro...", type=["jpg", "jpeg", "png"])
        
        # Check if the user wants to use a default image for testing (optional, not requested but good for quick check)
        # But per plan, just file uploader.

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Imagen cargada', width="stretch")
            
            with st.spinner('Analizando imagen...'):
                input_tensor = process_image(image).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # Get top 3 predictions
                top3_prob, top3_idx = torch.topk(probabilities, 3)
                
                st.subheader("Predicciones:")
                for i in range(3):
                    prob = top3_prob[i].item()
                    idx = top3_idx[i].item()
                    breed_key = class_names[idx]
                    
                    # Get translated name, fallback to original if not found
                    breed_name = translations.get(breed_key, breed_key.replace("_", " "))
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <span class="breed-name">{i+1}. {breed_name}</span>
                        <br>
                        <small>Confianza: {prob*100:.2f}%</small>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(prob)

    else:
        st.warning("Model could not be loaded. Please check the file paths.")
else:
    st.warning("Class names could not be loaded.")
