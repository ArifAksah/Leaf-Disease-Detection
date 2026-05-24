#Library imports
import numpy as np
import streamlit as st
import cv2
from tensorflow import keras

#Loading the Model
model = keras.models.load_model('CNNModel.keras')

#Name of Classes
CLASS_NAMES = ['Tomato-Early_Bright','Tomato-Healthy','Tomato-Late_bright',
               'Tomato-Leaf_Mold','Tomato-Septoria_LeafSpot',
               'Tomato-Spider_Mites','Tomato-Target_Spot',
               'Tomato-YellowLeaf-CurlVirus'
 'Tomato-Bacterial_spot',
 'Tomato-mosaic_virus']

#Setting Title of App
st.title("Tomato Leaf Disease Detection")
st.markdown("Upload an image of the plant leaf")

#Uploading the dog image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict Disease')

#On predict button click
if submit:

    if plant_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (224,224))
        
        #Convert image to 4 Dimension
        opencv_image.shape = (1,224,224,3)
        
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is "+result.split('-')[0]+ " leaf with " + result.split('-')[1]))