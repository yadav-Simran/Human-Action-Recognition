import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from django.core.wsgi import get_wsgi_application



st.title("Image Classification")
upload_file = st.sidebar.file_uploader("Upload Images", type="jpg")
generate_pred = st.sidebar.button("Predict")
model = tf.keras.models.load_model("HAR_model.h5")


def import_n_pred(image_data, model):
        size = (224, 224)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        reshape = img[np.newaxis, ...]
        pred = model.predict(reshape)
        return pred


if generate_pred:
        image = Image.open(upload_file)
        with st.expander("image", expanded=True):
            st.image(image, use_column_width=True)
        pred = import_n_pred(image, model)
        labels = [
            "calling",
            "hugging",
            "laughing",
            "texting",
            "using_laptop",
            "clapping",
            "drinking",
            "sleeping",
            "eating",
            "sitting",
            "running",
            "listening_to_music",
            "dancing",
            "cycling",
            "fighting",
        ]
        st.title("prediction of image is {}".format(labels[np.argmax(pred)]))
