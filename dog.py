import streamlit as st
import joblib
from PIL import Image
import numpy as np
import os
from skimage.feature import hog
from PIL import Image




model = joblib.load(r"D:\internship\SVM\model.sav")


st.title("Dog Breed Classifier")
st.write("Upload an image to classify it as a **cat** or **dog**!")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)

  
    temp_folder = 'temp'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    file_path = os.path.join(temp_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    
  
    def preprocess_image(image_path):
              img = Image.open(image_path).convert('L')
              img = img.resize((50, 50))
              img_array = np.array(img) / 255.0
              features, _ = hog(img_array, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
              return features.reshape(1, -1)



    img_array = preprocess_image(file_path)

  
    predicted_class = model.predict(img_array)[0]

    if predicted_class == 1:
        st.success("This is a **cat**!")
    else:
        st.success("This is a **dog**!")

    os.remove(file_path)

else:
    st.info("Upload a file to start.")

st.markdown(
    """
    ---
    Developed by Rafiya
    """
)
