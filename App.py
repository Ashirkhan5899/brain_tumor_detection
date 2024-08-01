from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
from torchvision import transforms
from cnnModel import CNNModel
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
# Instantiate your model
model = CNNModel()
# Load the trained model weights
model.load_state_dict(torch.load('Trained Model/cnn_trained_model.pth'))
model.eval()


# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.ToTensor()
])

# Define class labels
class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            probability = torch.softmax(output, dim=1).squeeze()[1].item()
            max_prob_class = class_labels[probabilities.argmax().item()]
            
        os.remove(filepath)
        return jsonify({'probability': probability, 'class':max_prob_class})



    #     with torch.no_grad():
    #         output = model(image)
    #         probabilities = torch.softmax(output, dim=1).squeeze()
    #         print("Raw output:", output)  # Debugging line
    #         print("Probabilities:", probabilities)  # Debugging line
    #         prob_values = probabilities.tolist()
    #         max_prob = max(prob_values)
    #         max_prob_class = class_labels[probabilities.argmax().item()]

    # return jsonify({'probability': max_prob, 'class': max_prob_class})

if __name__ == '__main__':
    app.run(debug=True)
