
import re
import streamlit as st
from PIL import Image

if __name__ == '__main__':
    title = 'Tomography simulator'
    st.set_page_config(page_title=title, layout='centered')
    st.title(title)
    detectors = st.sidebar.text_input("Number of detectors", 30)
    alpha_step = st.sidebar.text_input("\u0394\u03B1 step", 15)
    detectors_range = st.sidebar.text_input("Detectors range", 90)
    file = st.file_uploader('Upload an image')
    regex_pattern = r'[A-Z][a-z]+'

    if file is not None:
        image = Image.open(file)
        st.image(image, "Initial image")
        first_name = st.text_input("Patient's first name")
        last_name = st.text_input("Patient's last name")
        commentary = st.text_input("Medical commentary")
        if re.match(regex_pattern, first_name) and re.match(regex_pattern, last_name):
            if st.button("Save"):
                st.text("Saved!")
        else:
            st.warning("First name and last name should start with capital case and continue with non capital")
