from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)

model = load_model('HAR_model.h5')

def preprocess_image(file):
   
    file_bytes = file.read()

    img_stream = io.BytesIO(file_bytes)

    img = image.load_img(img_stream, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    return img_array


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    print(request.files)


    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img_array = preprocess_image(file)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        class_labels = [
            "calling", "hugging", "laughing", "texting", "using_laptop",
            "clapping", "drinking", "sleeping", "eating", "sitting",
            "running", "listening_to_music", "dancing", "cycling", "fighting",
        ]
        result = {'class': class_labels[predicted_class], 'confidence': float(prediction[0][predicted_class])}
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
