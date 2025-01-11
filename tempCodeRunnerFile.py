import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

# Initialize Flask application
app = Flask(__name__)

# Load the model
model = load_model('face_detection_model.keras')

# Set the image upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the file is an allowed image type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Index route to display the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image and make a prediction
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict using the model
        prediction = model.predict(img_array)
        
        # Check the prediction result
        result = 'Real' if np.argmax(prediction) == 0 else 'Fake'
        confidence = round(np.max(prediction) * 100, 2)
        
        return render_template('index.html', prediction=result, confidence=confidence, filename=filename)

    return render_template('index.html', error="Invalid file format")

# Run the app
if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
