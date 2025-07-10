import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import warnings
import logging
import traceback
import sys
import sklearn
from sklearn.compose import ColumnTransformer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ignorar advertencias de versiones
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Cargar el modelo actualizado
MODEL_PATH = 'modelo_enfermedades_actualizado.pkl'

# SOLUCIÓN: Parche para la clase faltante
try:
    # Intentar importar la clase desde el nuevo módulo
    from sklearn.compose._column_transformer import _RemainderColsList
    logger.info("Clase _RemainderColsList disponible en sklearn.compose._column_transformer")
except ImportError:
    # Definir manualmente la clase si no está disponible
    logger.warning("Definiendo manualmente _RemainderColsList")
    class _RemainderColsList(list):
        pass
    
    # Inyectar la clase en el módulo correcto
    import sklearn.compose._column_transformer
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
    logger.info("Inyectado _RemainderColsList en ColumnTransformer")

# Columnas requeridas por el modelo - ACTUALIZADAS
REQUIRED_COLUMNS = [
    'Hb_g/dL', 'Hto_%', 'Glóbulos_Rojos_mill/µL', 'VCM_fL', 'HDL_mg/dL', 
    'LDL_mg/dL', 'Glucosa_en_ayunas_mg/dL', 'Leucocitos_mil/µL', 
    'Plaquetas_mil/µL', 'VO2_máx_ml/kg/min', 'FC_máx_alcanzada_lpm', 
    'Presión_Arterial_Sistólica_mmHg', 'Presión_Arterial_Diastólica_mmHg',
    'IMC', 'CHCM_g/dL', 'Presión_sistólica_post_esfuerzo_mmHg', 
    'HRR_descenso_en_1_min', 'Cloruro_en_sudor_mmol/L', 'Volumen_de_sudor_mg',
    'Na_mmol/L', 'K_mmol/L', 'Colesterol_Total_mg/dL', 'Triglicéridos_mg/dL',
    'Genero', 'Edad', 'Estado'  # COLUMNAS ADICIONALES REQUERIDAS
]

# --- VERIFICACIÓN CRÍTICA DE CARGA ---
try:
    logger.info(f"Intentando cargar modelo desde {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        logger.critical(f"❌ ARCHIVO DE MODELO NO ENCONTRADO: {MODEL_PATH}")
        logger.critical("❌ Por favor, coloque el archivo del modelo en el directorio actual")
        sys.exit(1)
        
    model_artifacts = joblib.load(MODEL_PATH)
    model = model_artifacts['model']
    label_encoder = model_artifacts['label_encoder']
    accuracy = model_artifacts.get('accuracy', 0.8846)
    logger.info(f"✅ Modelo cargado exitosamente. Precisión: {accuracy:.4f}")
    
    # Verificación adicional: intentar una predicción simple con datos realistas
    try:
        # Crear datos de prueba más realistas
        dummy_data = pd.DataFrame({
            'Hb_g/dL': [14.5],
            'Hto_%': [42.0],
            'Glóbulos_Rojos_mill/µL': [4.8],
            'VCM_fL': [88.0],
            'HDL_mg/dL': [50.0],
            'LDL_mg/dL': [100.0],
            'Glucosa_en_ayunas_mg/dL': [95.0],
            'Leucocitos_mil/µL': [7.0],
            'Plaquetas_mil/µL': [250.0],
            'VO2_máx_ml/kg/min': [35.0],
            'FC_máx_alcanzada_lpm': [180.0],
            'Presión_Arterial_Sistólica_mmHg': [120.0],
            'Presión_Arterial_Diastólica_mmHg': [80.0],
            'IMC': [22.5],
            'CHCM_g/dL': [34.0],
            'Presión_sistólica_post_esfuerzo_mmHg': [130.0],
            'HRR_descenso_en_1_min': [20.0],
            'Cloruro_en_sudor_mmol/L': [25.0],
            'Volumen_de_sudor_mg': [100.0],
            'Na_mmol/L': [140.0],
            'K_mmol/L': [4.0],
            'Colesterol_Total_mg/dL': [190.0],
            'Triglicéridos_mg/dL': [120.0],
            'Genero': ['hombre'],   # COLUMNAS ADICIONALES
            'Edad': [35],           # CON VALORES REALISTAS
            'Estado': ['sano']      # 
        })
        
        prediction = model.predict(dummy_data)
        logger.info(f"✅ Verificación de predicción exitosa. Resultado: {prediction}")
    except Exception as e:
        logger.warning(f"⚠️ Advertencia en verificación de predicción: {str(e)}")
        logger.warning("El servicio continuará, pero verifique la compatibilidad del modelo")
except Exception as e:
    logger.critical(f"❌ ERROR FATAL AL CARGAR MODELO: {str(e)}")
    logger.critical(traceback.format_exc())
    logger.critical("❌ El servicio no puede iniciar sin el modelo. Terminando.")
    sys.exit(1)
# -------------------------------------

def preparar_datos(df):
    """Prepara los datos para la predicción asegurando todas las columnas requeridas"""
    # Asegurar que tenemos todas las columnas necesarias
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            # Asignar valores por defecto según el tipo de columna
            if col in ['Genero', 'Estado']:
                df[col] = 'desconocido'
            elif col == 'Edad':
                df[col] = 0
            else:
                df[col] = 0.0
            logger.warning(f"Columna requerida '{col}' no encontrada. Se añadió con valor por defecto.")
    
    # Convertir columnas numéricas
    numeric_cols = [col for col in REQUIRED_COLUMNS if col not in ['Genero', 'Estado']]
    for col in numeric_cols:
        # Manejar diferentes formatos numéricos
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Manejar columnas categóricas
    for col in ['Genero', 'Estado']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df[REQUIRED_COLUMNS].fillna(0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del request
        data = request.json
        
        # Si es una lista de registros, convertir a DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Formato de datos inválido. Se espera una lista de registros'}), 400
        
        # Preparar datos
        X = preparar_datos(df)
        
        # Realizar predicción
        y_pred_encoded = model.predict(X)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Calcular probabilidades
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
            confianza = np.max(y_proba, axis=1)
        else:
            confianza = np.ones(len(X))
        
        # Preparar respuesta
        result = df.copy()
        result['enfermedad_predicha'] = y_pred
        result['confianza_enfermedad'] = confianza
        
        # Calcular métricas
        distribucion = pd.Series(y_pred).value_counts().to_dict()
        
        return jsonify({
            'predictions': result.to_dict(orient='records'),
            'metrics': {
                'total_predicciones': len(df),
                'distribucion_enfermedades': distribucion,
                'model_accuracy': float(accuracy)
            }
        })
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'advice': 'Verifique el formato de los datos de entrada'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ready',
        'model_loaded': True,
        'model_accuracy': float(accuracy),
        'scikit_version': sklearn.__version__,
        'required_columns': REQUIRED_COLUMNS,
        'service': 'prediction-api'
    })

if __name__ == '__main__':
    logger.info(f"Versión de scikit-learn: {sklearn.__version__}")
    logger.info("✅ Servicio de predicción iniciado correctamente")
    app.run(port=5001, debug=True, threaded=True)