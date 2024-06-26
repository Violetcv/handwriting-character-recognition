from Flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import joblib

app = Flask(__name__)

# Load the handwriting recognition model
model = joblib.load('hr-clf.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('L')  # Convert to grayscale

    # Preprocess the image as needed for your model
    image = preprocess_image(image)  # This is a placeholder for your preprocessing function

    # Predict the class of the handwriting
    prediction = model.predict([image.flatten()])  # Flatten the image for prediction

    return jsonify({"prediction": bool(prediction[0])})

def preprocess_image(image):
    # Resize the image to 28x28 pixels if needed (assuming the model expects 28x28 input)
    image = image.resize((28, 28))
    
    # Convert the image to a numpy array and normalize pixel values (0-255 to 0-1)
    image = np.array(image) / 255.0
    
    return image

if __name__ == '__main__':
    app.run(debug=True)
