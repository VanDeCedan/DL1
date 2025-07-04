import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from src.main import parazitized_class_names,load_model_with_cache
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Replace with your cloud storage share link (OneDrive, Google Drive, or Dropbox)
try:
    PARAZITIZED_MODEL_URL = st.secrets['parazited_model_url ']
except:
    PARAZITIZED_MODEL_URL = os.getenv('parazited_model_url')

# Load the model (will download if not cached locally)
model_parasit = load_model_with_cache(PARAZITIZED_MODEL_URL)

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((50, 50))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image):
    image = preprocess_image(image)
    predictions = model_parasit.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return parazitized_class_names[predicted_class], predictions[0][predicted_class]

st.title("Classification d'images")

predictions = option_menu(
    menu_title=None,
    menu_icon="cast",
    options=["Images de globules rouges", "Chiens et chats"],
    icons=["moisture", "piggy-bank-fill"],
    orientation="horizontal"
)

if predictions == "Images de globules rouges":
    st.subheader("Classification des cellules infectées par le virus de la malaria")
    
    uploaded_file = st.file_uploader("Choisissez une image de cellule", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Image téléchargée', use_container_width=True)
        
        if st.button("Classer l'image"):
            label, confidence = predict(image)
            if label == 'cellules saines':
                st.success(f"Prédiction: {label} (Confiance: {confidence:.2f})", icon="✅")
            else:
                st.warning(f"Prédiction: {label} (Confiance: {confidence:.2f})",icon="⚠️")