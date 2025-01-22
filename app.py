# Description: This file contains the Flask application that serves the HTML page and handles the file upload and prediction.
# The model is loaded and used to make predictions on the uploaded images. The prediction result and confidence score are
# displayed on the HTML page. The app runs on the local server and can be accessed through a web browser.
#----------------------------------------------
import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

# Initialize Flask application
# Create an instance of the Flask class and set the name of the module to __name__.
app = Flask(__name__)

# Load the model
# Load the trained model using the load_model function from the keras.models module.
model = load_model('face_detection_model.keras')

# Set the image upload folder and allowed extensions
# Set the upload folder for the images and the allowed extensions for the uploaded files.
UPLOAD_FOLDER = 'static/uploads'
# Define the allowed extensions for the uploaded files.
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# Set the upload folder for the images.
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the file is an allowed image type
def allowed_file(filename):
    # Check if the file extension is in the set of allowed extensions.
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Index route to display the HTML page
# Define the index route to display the HTML page.
@app.route('/')
def index():
    # Render the index.html template using the render_template function from the Flask module.
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
# Define the predict route to handle the file upload and make predictions using the model.
def predict():
    # Check if the file is uploaded
    if 'file' not in request.files:
        # Return an error message if no file part is found in the request.
        return render_template('index.html', error="No file part")
    # Get the file from the request
    file = request.files['file']
    # Check if the file is empty
    if file.filename == '':
        # Return an error message if no file is selected.
        return render_template('index.html', error="No selected file")
    # Check if the file is an allowed image type
    if file and allowed_file(file.filename):
        # Save the uploaded file to the upload folder
        filename = secure_filename(file.filename)
        # Save the uploaded file to the upload folder with the secure filename.
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Save the uploaded file to the upload folder.
        file.save(filepath)
        
        # Preprocess the image and make a prediction
        img = image.load_img(filepath, target_size=(224, 224))
        # Preprocess the image using the load_img function from the keras.preprocessing.image module.
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        # Expand the dimensions of the image array using the expand_dims function from the numpy module.
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict using the model
        prediction = model.predict(img_array)
        
        # Check the prediction result
        result = 'Real' if np.argmax(prediction) == 0 else 'Fake'
        # Calculate the confidence score
        confidence = round(np.max(prediction) * 100, 2)
        # Return the prediction result and confidence score to the HTML page.
        return render_template('index.html', prediction=result, confidence=confidence, filename=filename)
# Return an error message if the file format is not allowed.
    return render_template('index.html', error="Invalid file format")
# Define the main block to run the app.
# Run the app
if __name__ == "__main__":
    # Check if the upload folder exists and create it if it does not.
    if not os.path.exists(UPLOAD_FOLDER):
        # Create the upload folder if it does not exist.
        os.makedirs(UPLOAD_FOLDER)
        # Run the app on the local server with debug mode enabled.
    app.run(debug=True)
    # Run the app on the local server with debug mode enabled.
