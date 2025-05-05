from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os
import ssl
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

# Disable SSL verification for downloading pre-trained models, if needed
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# Load the pretrained model
model = tf.keras.models.load_model('Model/Blood_group_detection.h5')

# Define the allowed extensions for upload files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path):
    """
    Preprocess the image for model predictions.

    Args:
        numpy.ndarray: Preprocess image ready to prediction.
    """
    # Load the image
    img = load_img(file_path, target_size = (64, 64)) # Resize to match the model's input size
    img_array = img_to_array(img) # Convert image to array
    img_array = np.expand_dims(img_array, axis = 0) # Add batch dimension
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

# Endpoint to predict the blood group from fingerprint image
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file Provided'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file Selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types are png, jpg, jpeg'}), 400
    
    # Save the uploaded files
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    try:
        # Preprocess the image
        img = preprocess_image(file_path)

        # Perform prediction
        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions[0]))
        print('Predicted class is: ', predicted_class)

        # Optional: define class names (if not in the model)
        class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        predicted_label = class_names[predicted_class]

        # Return the result as JSON
        return jsonify({
            'Predicted_class': predicted_class,
            'Predicted_label': predicted_label,
            'confidence': float(np.argmax(predictions[0]))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    #finally:
        # Clean up: Remove the saved files
    #    if os.path.exists(file_path):
    #        os.remove(file_path)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)