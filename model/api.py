from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf  
from io import BytesIO

app = Flask(__name__)

#Bottle Model
model_bottle_size = tf.keras.models.load_model('bottle_size_model.h5') 
model_bottle_brand = tf.keras.models.load_model('bottle_brand_model.h5')  

bottle_size_classes = ['bottel_1600', 'bottel_350', 'bottle_1250', 'bottle_1500', 'bottle_1950', 'bottle_280', 'bottle_300', 'bottle_320', 'bottle_322', 'bottle_340', 'bottle_360', 'bottle_400', 'bottle_410', 'bottle_430', 'bottle_440', 'bottle_445', 'bottle_500', 'bottle_600ml', 'bottle_640', 'bottle_750'] 
bottle_brand_classes = ['amphawa', 'amwelplus', 'aquafina', 'beauti_drink', 'big', 'coca_cola', 'cocomax', 'crystal', 'est', 'ichitan', 'kato', 'mansome', 'mikko', 'minearlwater', 'nestle', 'no_band', 'oishi', 'pepsi', 'sing', 'spinking_water', 'sprite', 'srithep', 'tipchumporn_drinking_water'] 

#Can Model
model_can_size = tf.keras.models.load_model('can_size_model.h5') 
model_can_brand = tf.keras.models.load_model('can_brand_model.h5')  

can_size_classes = ['can_180', 'can_245', 'can_330', 'can_490'] 
can_brand_classes = ['birdy', 'calpis_lacto', 'chang', 'green_mate', 'leo', 'nescafe', 'sing'] 

CONFIDENCE_THRESHOLD = 0.5

def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    image = image.resize((256, 144))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

def predict_category(image_bytes, model_size, model_brand, size_classes, brand_classes):
    processed_image = preprocess_image(image_bytes)

    size_prediction = model_size.predict(processed_image)
    size_index = np.argmax(size_prediction)  
    size_confidence = np.max(size_prediction) 

    brand_prediction = model_brand.predict(processed_image)
    brand_index = np.argmax(brand_prediction)  
    brand_confidence = np.max(brand_prediction)  

    size_name = size_classes[size_index] if size_confidence >= CONFIDENCE_THRESHOLD else "unknown"
    brand_name = brand_classes[brand_index] if brand_confidence >= CONFIDENCE_THRESHOLD else "unknown"

    return size_name, brand_name

@app.route('/predict/bottle', methods=['POST'])
def predict_bottle():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        size_name, brand_name = predict_category(file.read(), model_bottle_size, model_bottle_brand, bottle_size_classes, bottle_brand_classes)
        return jsonify({
            "bottle_size": size_name,  
            "bottle_brand": brand_name  
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/can', methods=['POST'])
def predict_can():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        size_name, brand_name = predict_category(file.read(), model_can_size, model_can_brand, can_size_classes, can_brand_classes)
        return jsonify({
            "can_size": size_name,  
            "can_brand": brand_name  
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
