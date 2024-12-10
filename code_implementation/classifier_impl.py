import os
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array

model = tf.keras.models.load_model('models/good_bad_classifier.h5')

image_path = 'path_to_your_image.jpg'  
image_size = (150, 150)  

# Preprocess the image
image = load_img(image_path, target_size=image_size)  
image_array = img_to_array(image)  
image_array = image_array / 255.0  
image_array = np.expand_dims(image_array, axis=0)  

# Predict using the model
prediction = model.predict(image_array)
predicted_label = 'Bad' if prediction[0] > 0.5 else 'Good'

# Output results
print(f"Prediction Confidence: {prediction[0][0]:.2f}")
print(f"Predicted Class: {predicted_label}")