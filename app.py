import os
import glob

import streamlit as st
from main import predict
from PIL import Image

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

# all models list
models_list = [
            "vgg16",
            "vgg19",
            "inception",
            "xception",
            "resnet"
        ]

# all images in 'images' folder

uploaded_image = st.sidebar.file_uploader("Upload your own image", type=["jpg"])
if uploaded_image is not None:
    uploaded_image = Image.open(uploaded_image)
    uploaded_image_show = uploaded_image.resize((224,224))
    st.sidebar.image(uploaded_image_show, use_column_width=True)
# print(uploaded_image)
images = [image.split("/")[1] for image in glob.glob(os.path.join("images", "*.jpg"))]
selected_image = st.sidebar.selectbox("Pick an image.", images)

st.sidebar.title("ImageNet")
selected_model = st.sidebar.selectbox("Pick a model.", models_list)

st.write("Enjoy Machine Learning with Streamlit !!!")

if st.sidebar.button('Predict'):
    showpred = 1
    inputShape = (224, 224)
    if selected_model in ["inception", "xception"]:
        inputShape = (299, 299)
    if uploaded_image is None:
        print('[$$$$] preset image')
        image_path = os.path.join("images", selected_image)
        print(f"[#] processing {image_path} ... ", end="")
        image_show = load_img(image_path, target_size=inputShape)
    else:
        image_show = uploaded_image.resize(inputShape)
    image = img_to_array(image_show)
    print("Done")

    prediction, prob = predict(image, selected_model)
    # image = image.resize((128, 128))
    # st.write(prediction)
    st.image(image_show,
             caption=f"prediction: {prediction}, probability: {prob * 100}",
             use_column_width=True
    )