import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

st.title('Potato Leaf Disease Detection')

def leaf_detected(image):
    # Add your logic here to detect if a leaf is present in the image.
    # If a leaf is detected, return True; otherwise, return False.
    # Replace this placeholder logic with your actual implementation.
    return True

def main():
    file_uploaded = st.file_uploader('Choose an image...', type='jpg')
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        
        # Detect the plant in the image.
        plant = detect_plant(image)

        # Check if the plant is a potato leaf.
        if plant.is_potato_leaf():
            result, confidence = predict_class(image)
            st.write('PREDICTION: {}'.format(result))
            st.write('CONFIDENCE: {}%'.format(confidence))
        else: 
            st.warning("No potato leaf detected. Please retake the image.")
            st.button("Retake Image", key="retake_button")

def detect_plant(image):
    # Add your logic here to detect the plant in the image.
    # If a specific plant is detected, return an object representing it;
    # otherwise, return None.
    # Replace this placeholder logic with your actual implementation.
    return DetectedPlant()

class DetectedPlant:
    def is_potato_leaf(self):
        # Add your logic here to determine if the detected plant is a potato leaf.
        # If it is, return True; otherwise, return False.
        # Replace this placeholder logic with your actual implementation.
        return True

def predict_class(image):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="C:/Users/deepr/Downloads/model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape'][1:3]

    test_image = image.resize(input_shape)
    test_image = np.array(test_image)
    test_image = (test_image / 255.0).astype('float32')

    # Prepare the input tensor
    input_data = np.expand_dims(test_image, axis=0)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(output_data)
    confidence = output_data[0][predicted_class_index] * 100.0

    class_names = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___healthy',
 'Strawberry___Leaf_scorch',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___healthy',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']  # Add your class names

    final_pred = class_names[predicted_class_index]
    return final_pred, confidence

if __name__ == '__main__':
    main()
