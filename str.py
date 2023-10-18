import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="E:/sk/skin_cancer_model.tflite")
interpreter.allocate_tensors()

# Mapping of class indices to class names
class_names = {
    0: "Melanocytic nevi",
    1: "Melanoma",
    2: "Benign keratosis-like lesions",
    3: "Basal cell carcinoma",
    4: "Actinic keratoses",
    5: "Vascular lesions",
    6: "Dermatofibroma"
}

# Define your Streamlit app
def main():
    st.title("Skin Disease Prediction App")
    st.write("Upload an image for prediction")

    # Image uploader
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Preprocess the image
        img = Image.open(uploaded_image)
        img = img.resize((128, 128))  # Resize the image to the input size expected by the model
        input_data = np.expand_dims(img, axis=0).astype(np.float32)  # Convert the image to a numpy array

        # Perform inference
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(output_data)
        predicted_class = class_names[predicted_class_index]
        predicted_probability = output_data[0][predicted_class_index]
        percentage = round(predicted_probability * 100, 2)

        # Display results
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write(f"Prediction: {predicted_class}")

if __name__ == '__main__':
    main()
