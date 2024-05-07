import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from predicted_classes import get_class_name

# Load your trained model
cnn_model = tf.keras.models.load_model("./models/cnn.keras")
ann_model = tf.keras.models.load_model("./models/ann.keras")


def process_image(image):
    img = image.resize((30, 30))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.utils.normalize(img_array, axis=1)
    return img_array


def predict_image(image, model_name):
    if model_name == "CNN":
        model = cnn_model
    else:
        model = ann_model
    processed_image = process_image(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    class_name = get_class_name(predicted_class_index)
    return predicted_class_index, class_name


def main():
    st.title("German Traffic Sign Classification App")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    model_option = st.selectbox("Choose Model", ["CNN", "ANN"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            predicted_label, class_name = predict_image(image, model_option)
            st.write(f"Prediction Class: ({model_option}): ", predicted_label)
            st.write(f"Prediction Class Name: ({model_option}): ", class_name)


if __name__ == "__main__":
    main()
