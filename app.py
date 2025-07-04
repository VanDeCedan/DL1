import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from src.main import predict


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