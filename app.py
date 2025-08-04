from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

# Variables globales
model = None
scaler = None
encoder_mappings = None

def load_models():
    global model, scaler, encoder_mappings
    try:
        model = joblib.load('models/rf_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/encoder_mappings.json', 'r', encoding='utf-8') as f:
            encoder_mappings = json.load(f)
        print("✅ Modelos cargados correctamente")
    except Exception as e:
        print(f"❌ Error cargando modelos: {str(e)}")
        raise

load_models()

def get_encoded_value(category: str, column: str) -> int:
    """Obtiene el valor codificado para una categoría"""
    try:
        categories = encoder_mappings['column_categories'][column]['categories']
        return categories.index(category) if category in categories else 0
    except KeyError:
        return 0

def get_estimated_weight(presentation: str) -> float:
    """Calcula el peso estimado basado en la presentación"""
    weight_map = {
        '1kg': 1000, '2lb': 907, '500g': 500, '1L': 1000,
        '250mL': 250, '30 porciones': 900, '120 cápsulas': 120,
        'pack x12': 1200, 'servicio x1': 1000
    }
    return weight_map.get(presentation, 1000)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validación de campos requeridos (ahora sin 'status')
        required_fields = ['stock', 'category', 'brand', 'presentation']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Faltan campos requeridos'}), 400

        # Calcular status automáticamente
        stock = float(data['stock'])
        if stock == 0:
            status = 'OUT_OF_STOCK'
        elif stock <= 5:
            status = 'LOW_STOCK'
        else:
            status = 'STOCK'

        # Crear dataframe con todas las características
        input_data_full = {
            'stock': stock,
            'status': get_encoded_value(status, 'status'),  # Calculado automáticamente
            'category': get_encoded_value(data['category'], 'category'),
            'brand': get_encoded_value(data['brand'], 'brand'),
            'presentation': get_encoded_value(data['presentation'], 'presentation'),
            'peso_estimado': get_estimated_weight(data['presentation']),
            'marca_premium': 1 if data['brand'] in ['Optimum Nutrition', 'BSN', 'Dymatize'] else 0
        }

        # Resto del proceso permanece igual
        feature_order_full = ['stock', 'status', 'category', 'brand', 'presentation', 'peso_estimado', 'marca_premium']
        input_array_full = np.array([[input_data_full[field] for field in feature_order_full]])
        scaled_input = scaler.transform(input_array_full)
        
        selected_features = ['peso_estimado', 'presentation', 'category', 'status', 'marca_premium']
        feature_indices = [feature_order_full.index(f) for f in selected_features]
        input_array_model = scaled_input[:, feature_indices]
        
        prediction = model.predict(input_array_model)
        
        return jsonify({
            'success': True,
            'predictedPrice': round(float(prediction[0]), 2),
            'currency': 'MXN',
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error processing prediction'
        }), 500

if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)