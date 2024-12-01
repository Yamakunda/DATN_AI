from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],  # React development server
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load model and initialize data structures
model_data = tf.keras.models.load_model('./DogSymptomsModel/dog_model.h5')
model_image = tf.keras.models.load_model('./SkinDiseaseModel/SkinDisease.h5')
df = pd.read_csv('./dataset/pre_processed.csv')
data = df.drop(['Unnamed: 0', 'Disease'], axis=1)

# Define X and y columns
disease_columns = ['Tick fever', 'Distemper', 'Parvovirus',
       'Hepatitis', 'Tetanus', 'Chronic kidney Disease', 'Diabetes',
       'Gastrointestinal Disease', 'Allergies', 'Gingitivis', 'Cancers',
       'Skin Rashes']

X = data.drop(disease_columns, axis=1)
y = data[disease_columns]

@app.route('/predict/data', methods=['POST'])
def predictData():
    try:
        # Get input symptoms from request
        content = request.json
        input_labels = content['symptoms']

        # Create empty DataFrames
        X_drop = X.iloc[0:0]
        Y_drop = y.iloc[0:0]

        # Process input symptoms
        new_row_x = {col: 1 if col in input_labels else 0 for col in X_drop.columns.tolist()}
        new_row_df = pd.DataFrame([new_row_x])
        X_drop = pd.concat([X_drop, new_row_df], ignore_index=True)

        # Make prediction
        predict = model_data.predict(X_drop)
        
        # Process prediction results
        new_row_y = pd.DataFrame(predict, columns=Y_drop.columns)
        Y_drop = pd.concat([Y_drop, new_row_y], ignore_index=True)

        # Convert predictions to dictionary
        predictions = Y_drop.iloc[0].to_dict()

        return jsonify({
            "success": True,
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@app.route('/predict/image', methods=['POST'])
def predictImage():
    try:
        # Get JSON data from request body
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No base64 image provided'}), 400
        
        # Get base64 string
        base64_image = data['image']
        
        # Remove base64 prefix if exists
        if 'data:image' in base64_image:
            base64_image = base64_image.split(',')[1]
            
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_image)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((150, 150))
        
        # Predict using existing function
        predict_product, df = predict_image(image)
        
        # Convert DataFrame to dictionary
        probabilities = df.to_dict(orient='records')
        
        return jsonify({
            'prediction': predict_product,
            'probabilities': probabilities
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_image(image_upload, model=model_image):
    im_array = np.asarray(image_upload)
    im_array = im_array*(1.0/225.)
    im_input = tf.reshape(im_array, shape=[1, 150, 150, 3])

    predict_array = model.predict(im_input)[0]

    import pandas as pd
    df = pd.DataFrame(predict_array)
    df = df.rename({0:'Probability'}, axis='columns')
    prod = ['flea_allergy', 'hotspot', 'mange', 'ringworm']
    df['Animal'] = prod
    df = df[['Animal', 'Probability']]

    predict_label = np.argmax(model.predict(im_input))

    if predict_label == 0:
        predict_product = 'flea_allergy'
    elif predict_label == 1:
        predict_product = 'hotspot'
    elif predict_label == 2:
        predict_product = 'mange'
    else:
        predict_product = 'ringworm'

    return predict_product, df

if __name__ == '__main__':
    app.run(debug=True)