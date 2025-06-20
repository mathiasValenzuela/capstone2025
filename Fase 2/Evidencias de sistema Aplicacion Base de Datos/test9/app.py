from flask import Flask, render_template, request, jsonify, redirect, url_for, g
from pymongo import MongoClient
import numpy as np
import json
import re
import requests
import pandas as pd
import joblib
import xgboost as xgb
from datetime import datetime, timedelta
from bson import ObjectId
from bson.errors import InvalidId
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

app = Flask(__name__)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Conexión a la base de datos MongoDB
client = MongoClient('mongodb://localhost:27017/', connectTimeoutMS=30000, socketTimeoutMS=30000)
db = client['MEDICALANALYTICS1']

# Manejo del modelo XGBoost
modelo_xgb = None
try:
    if os.path.exists('modelo_xgboost.pkl'):
        modelo_xgb = joblib.load('modelo_xgboost.pkl')
        logger.info("Modelo XGBoost cargado exitosamente")
    else:
        logger.warning("Archivo 'modelo_xgboost.pkl' no encontrado. Continuando sin modelo de predicción.")
except Exception as e:
    logger.error(f"Error cargando modelo XGBoost: {str(e)}")

# Configuración para el modelo de lenguaje local (Ollama)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

# Lista de meses para los selectores de la UI
MESES = [
    'ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO',
    'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE'
]

# Indicadores clave para el dashboard
INDICADORES_CLAVE = [
    'Hb_g/dL', 'Hto_%', 'Glóbulos_Rojos_mill/µL', 'VCM_fL', 'HDL_mg/dL', 'LDL_mg/dL',
    'Glucosa_en_ayunas_mg/dL', 'Leucocitos_mil/µL', 'Plaquetas_mil/µL', 'VO2_máx_ml/kg/min',
    'FC_máx_alcanzada_lpm', 'Presión_Arterial_Sistólica_mmHg', 'Presión_Arterial_Diastólica_mmHg',
    'IMC', 'CHCM_g/dL', 'Presión_sistólica_post_esfuerzo_mmHg', 'HRR_descenso_en_1_min',
    'Cloruro_en_sudor_mmol/L', 'Volumen_de_sudor_mg', 'Na_mmol/L', 'K_mmol/L',
    'Colesterol_Total_mg/dL', 'Triglicéridos_mg/dL'
]

# Rangos de referencia saludables
HEALTH_RANGES = {
    'Hb_g/dL': {'hombre': (13, 17), 'mujer': (12, 15)},
    'Hto_%': {'hombre': (40, 50), 'mujer': (36, 44)},
    'Glóbulos_Rojos_mill/µL': {'hombre': (4.7, 6.1), 'mujer': (4.2, 5.4)},
    'VCM_fL': (80, 100),
    'HDL_mg/dL': {'hombre': (40, float('inf')), 'mujer': (50, float('inf'))},
    'LDL_mg/dL': (float('-inf'), 100),
    'Glucosa_en_ayunas_mg/dL': (70, 99),
    'Leucocitos_mil/µL': (4.0, 11.0),
    'Plaquetas_mil/µL': (150, 450),
    'Presión_Arterial_Sistólica_mmHg': (float('-inf'), 120),
    'Presión_Arterial_Diastólica_mmHg': (float('-inf'), 80),
    'IMC': (18.5, 24.9),
    'CHCM_g/dL': (32, 36),
    'Cloruro_en_sudor_mmol/L': (float('-inf'), 30),
    'Na_mmol/L': (135, 145),
    'K_mmol/L': (3.5, 5.0),
    'Colesterol_Total_mg/dL': (float('-inf'), 200),
    'Triglicéridos_mg/dL': (float('-inf'), 150)
}

def calcular_priorizacion(pacientes_data):
    if not pacientes_data:
        return []

    processed_pacientes = []
    for p in pacientes_data:
        coeficiente_error = 0
        genero = p.get('Genero', 'hombre').lower()
        for key, ranges in HEALTH_RANGES.items():
            try:
                value_str = str(p.get(key, '0')).replace(',', '.')
                value = float(value_str)
            except (ValueError, TypeError):
                continue

            current_range = ranges.get(genero) if isinstance(ranges, dict) else ranges
            if not current_range: continue

            min_val, max_val = current_range
            deviation = 0
            
            if value < min_val:
                normalizer = min_val if min_val != 0 else 1
                deviation = (value - min_val) / normalizer
            elif value > max_val and max_val != float('inf'):
                normalizer = max_val if max_val != 0 else 1
                deviation = (value - max_val) / normalizer
            
            coeficiente_error += abs(deviation) * 100
        
        p['coeficiente_error'] = round(coeficiente_error, 2)
        processed_pacientes.append(p)

    max_coeficiente = max((p.get('coeficiente_error', 0) for p in processed_pacientes), default=0)

    for p in processed_pacientes:
        p['priorizacion'] = round((p.get('coeficiente_error', 0) / max_coeficiente) * 100, 2) if max_coeficiente > 0 else 0
            
    return processed_pacientes

def predecir_con_xgboost(df, mes):
    if modelo_xgb is None:
        return None

    try:
        df['target'] = (df['Estado'] != 'sano').astype(int)
        
        features_a_eliminar = [
            '_id', 'PacienteID', 'Nombre', 'FechaNacimiento', 'FechaAtencion',
            'Estado', 'Enfermedades', 'target', 'priorizacion', 'coeficiente_error'
        ]
        columnas_existentes_a_eliminar = [col for col in df.columns if col in features_a_eliminar]
        
        X = df.drop(columns=columnas_existentes_a_eliminar)
        y = df['target']

        y_pred = modelo_xgb.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        report = classification_report(y, y_pred, target_names=['Sano', 'Con Patología'], output_dict=True, zero_division=0)
        cm = confusion_matrix(y, y_pred).tolist()
        
        return {'accuracy': accuracy, 'report': report, 'confusion_matrix': cm, 'mes': mes, 'total_datos': len(df)}
    except Exception as e:
        logger.error(f"Error en la predicción con XGBoost: {str(e)}")
        return None

@app.route('/')
def index():
    mes = request.args.get('mes', 'ENERO')
    coleccion = db[mes]
    pacientes_data = list(coleccion.find().limit(50))
    pacientes_con_priorizacion = calcular_priorizacion(pacientes_data)
    pacientes_con_priorizacion.sort(key=lambda x: x.get('priorizacion', 0), reverse=True)
    return render_template('index.html', pacientes=pacientes_con_priorizacion, meses=MESES, mes_actual=mes)

@app.route('/search')
def search():
    query = request.args.get('query', '')
    mes = request.args.get('mes', 'ENERO')
    coleccion = db[mes]
    regex = re.compile(f'.*{re.escape(query)}.*', re.IGNORECASE)
    pacientes_data = list(coleccion.find({'$or': [{'Nombre': regex}, {'RUT': regex}, {'Estado': regex}]}).limit(50))
    pacientes_con_priorizacion = calcular_priorizacion(pacientes_data)
    pacientes_con_priorizacion.sort(key=lambda x: x.get('priorizacion', 0), reverse=True)
    return render_template('index.html', pacientes=pacientes_con_priorizacion, meses=MESES, mes_actual=mes)

@app.route('/dashboard')
def dashboard():
    mes = request.args.get('mes', 'ENERO')
    coleccion = db[mes]
    total_pacientes = coleccion.count_documents({})
    
    # --- Datos para Gráficos ---

    # Top 5 enfermedades
    pipeline_enfermedades = [
        {"$unwind": "$Enfermedades"},
        {"$match": {"Enfermedades": {"$nin": ["Ninguna", "ninguna", "Sano", "sano"]}}},
        {"$group": {"_id": "$Enfermedades", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5}
    ]
    top_enfermedades_data = list(coleccion.aggregate(pipeline_enfermedades))
    top_enfermedades_labels = [e['_id'] for e in top_enfermedades_data]
    top_enfermedades_values = [e['count'] for e in top_enfermedades_data]

    # Priorización de pacientes (para gráfico de burbujas y barras)
    pacientes_data_all = list(coleccion.find())
    pacientes_con_priorizacion = calcular_priorizacion(pacientes_data_all)
    pacientes_con_priorizacion.sort(key=lambda x: x.get('priorizacion', 0), reverse=True)
    
    # Datos para gráfico de barras de priorización (Top 10)
    priorizacion_labels = [p.get('Nombre', 'N/A') for p in pacientes_con_priorizacion[:10]]
    priorizacion_values = [p.get('priorizacion', 0) for p in pacientes_con_priorizacion[:10]]

    # Estado de salud
    pipeline_estado = [{"$group": {"_id": "$Estado", "count": {"$sum": 1}}}]
    estado_data = list(coleccion.aggregate(pipeline_estado))
    estado_labels = [e.get('_id', 'N/A') for e in estado_data]
    estado_values = [e.get('count', 0) for e in estado_data]
    
    # Distribución por Género
    pipeline_genero = [{"$group": {"_id": "$Genero", "count": {"$sum": 1}}}]
    genero_data = list(coleccion.aggregate(pipeline_genero))
    genero_labels = [g.get('_id', 'N/A') for g in genero_data]
    genero_values = [g.get('count', 0) for g in genero_data]
    
    # Datos para gráfico de burbujas (Priorización vs Edad vs IMC)
    bubble_chart_data = []
    for p in pacientes_con_priorizacion:
        try:
            edad = float(p.get('Edad'))
            priorizacion = float(p.get('priorizacion'))
            imc = float(p.get('IMC', 0))
            # El radio de la burbuja debe ser positivo
            bubble_chart_data.append({'x': edad, 'y': priorizacion, 'r': max(imc / 2, 5)})
        except (TypeError, ValueError):
            continue

    # Datos para gráfico de dispersión (Glucosa vs Edad)
    scatter_glucosa_data = []
    for p in pacientes_data_all:
        try:
            edad = float(p.get('Edad'))
            glucosa_str = str(p.get('Glucosa_en_ayunas_mg/dL', '0')).replace(',', '.')
            glucosa = float(glucosa_str)
            scatter_glucosa_data.append({'x': edad, 'y': glucosa})
        except (TypeError, ValueError):
            continue

    # --- Predicción con Modelo ML ---
    datos_prediccion = None
    if modelo_xgb and pacientes_data_all:
        try:
            pacientes_df = pd.DataFrame(pacientes_data_all)
            if not pacientes_df.empty:
                for col in INDICADORES_CLAVE:
                    if col in pacientes_df.columns:
                        pacientes_df[col] = pd.to_numeric(pacientes_df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
                pacientes_df.fillna(0, inplace=True)
                datos_prediccion = predecir_con_xgboost(pacientes_df.copy(), mes) 
        except Exception as e:
            logger.error(f"Error preparando datos para predicción: {str(e)}")

    modelos_recomendados = [
        {"nombre": "XGBoost", "precision": 92, "datos_procesados": ">1M registros", "ventajas": "Alto rendimiento con grandes datasets, manejo eficiente de variables faltantes"},
        {"nombre": "Random Forest", "precision": 89, "datos_procesados": "500K registros", "ventajas": "Robusto contra overfitting, manejo de características no lineales"},
        {"nombre": "Redes Neuronales", "precision": 91, "datos_procesados": "2M registros", "ventajas": "Captura relaciones complejas, buen rendimiento con datos estructurados"},
        {"nombre": "LightGBM", "precision": 93, "datos_procesados": "1.5M registros", "ventajas": "Entrenamiento rápido, eficiente con grandes volúmenes de datos"}
    ]

    return render_template('dashboard.html', 
        total_pacientes=total_pacientes, 
        top_enfermedades_labels=json.dumps(top_enfermedades_labels), 
        top_enfermedades_values=json.dumps(top_enfermedades_values), 
        meses=MESES, 
        mes_actual=mes, 
        prediccion=datos_prediccion,
        priorizacion_labels=json.dumps(priorizacion_labels),
        priorizacion_values=json.dumps(priorizacion_values),
        estado_labels=json.dumps(estado_labels),
        estado_values=json.dumps(estado_values),
        genero_labels=json.dumps(genero_labels),
        genero_values=json.dumps(genero_values),
        bubble_chart_data=json.dumps(bubble_chart_data),
        scatter_glucosa_data=json.dumps(scatter_glucosa_data),
        modelos_recomendados=modelos_recomendados
    )

@app.route('/agendar')
def agendar():
    return render_template('agendar.html')

@app.route('/asistencia')
def asistencia():
    return render_template('asistencia.html')

@app.route('/editar/<mes>/<id_paciente>')
def editar_paciente(mes, id_paciente):
    try:
        paciente = db[mes].find_one({"_id": ObjectId(id_paciente)})
        return render_template('formulario.html', paciente=paciente, mes=mes) if paciente else ("Paciente no encontrado", 404)
    except InvalidId:
        return "ID de paciente inválido", 400

@app.route('/actualizar/<mes>/<id_paciente>', methods=['POST'])
def actualizar_paciente(mes, id_paciente):
    try:
        datos_actualizados = {}
        for key, value in request.form.items():
            if value != '':
                if key not in ['Nombre', 'RUT', 'Genero', 'Estado', 'Enfermedades']:
                    try:
                        datos_actualizados[key] = float(str(value).replace(',', '.'))
                    except ValueError:
                        datos_actualizados[key] = value
                else:
                    datos_actualizados[key] = value
        
        db[mes].update_one({"_id": ObjectId(id_paciente)}, {"$set": datos_actualizados})
        return redirect(url_for('index', mes=mes))
    except Exception as e:
        logger.error(f"Error al actualizar paciente: {str(e)}")
        return f"Error al actualizar: {str(e)}", 500

def buscar_paciente_en_mes(month, query, search_type):
    try:
        logger.info(f"Buscando en {month} - Tipo: {search_type} - Query: {query}")
        coleccion = db[month]
        
        if search_type == 'RUT':
            rut = query.replace('.', '').replace('-', '').upper()
            regex = re.compile(f'^{re.escape(rut)}.*', re.IGNORECASE)
            pacientes = list(coleccion.find({'RUT': regex}))
        elif search_type == 'Nombre':
            regex = re.compile(f'.*{re.escape(query)}.*', re.IGNORECASE)
            pacientes = list(coleccion.find({'Nombre': regex}))
        else:
            regex = re.compile(f'.*{re.escape(query)}.*', re.IGNORECASE)
            pacientes = list(coleccion.find({'$or': [{'Nombre': regex}, {'RUT': regex}]}))
        
        logger.info(f"Encontrados {len(pacientes)} pacientes en {month}")
        return [(month, paciente) for paciente in pacientes]
    except Exception as e:
        logger.error(f"Error buscando en {month}: {str(e)}")
        return []

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    data = request.json
    user_message = data.get('mensaje')
    logger.info(f"Consulta recibida: {user_message}")
    
    rut_match = re.search(r'(\d{1,2}\.?\d{3}\.?\d{3}-?[\dkK])', user_message, re.IGNORECASE)
    name_match = re.search(r'(?:paciente|nombre|name|buscar|información|datos|historial)[:\s]*([A-Za-zÁÉÍÓÚáéíóúñÑ\s]+)', user_message, re.IGNORECASE)
    
    specified_month = None
    for month in MESES:
        if month.lower() in user_message.lower():
            specified_month = month
            break
    
    months_to_search = [specified_month] if specified_month else MESES
    patient_info_context = ""
    found_patients = []
    search_type = ""
    search_query = ""

    if rut_match:
        search_query = rut_match.group(1)
        search_type = "RUT"
        logger.info(f"Búsqueda por RUT detectada: {search_query}")
    elif name_match:
        search_query = name_match.group(1).strip()
        search_type = "Nombre"
        logger.info(f"Búsqueda por nombre detectada: {search_query}")
    else:
        search_query = user_message
        search_type = "General"
        logger.info(f"Búsqueda general: {search_query}")
    
    if months_to_search:
        with ThreadPoolExecutor(max_workers=min(4, len(months_to_search))) as executor:
            futures = [executor.submit(buscar_paciente_en_mes, month, search_query, search_type) for month in months_to_search]
            for future in as_completed(futures):
                if result := future.result():
                    found_patients.extend(result)
    
    if found_patients:
        logger.info(f"Total pacientes encontrados: {len(found_patients)}")
        for month, paciente in found_patients:
            patient_info_context += f"--- Paciente encontrado en {month} ---\n"
            for key, value in paciente.items():
                if key != '_id':
                    patient_info_context += f"- {key}: {value}\n"
            patient_info_context += "\n"
    else:
        patient_info_context = "⚠️ No se encontraron pacientes con los datos proporcionados."
        logger.warning("No se encontraron pacientes")
    
    prompt_for_ollama = (
        "Eres un asistente médico experto. El usuario ha solicitado información sobre pacientes. "
        f"Contexto de la base de datos:\n{patient_info_context}\n\n"
        f"Pregunta del usuario: {user_message}\n\n"
        "Responde de manera completa y profesional. Si el usuario pregunta sobre términos médicos, "
        "explica de forma simple pero precisa. Para datos numéricos, indica si están dentro de rangos saludables."
    )

    payload = {"model": MODEL_NAME, "prompt": prompt_for_ollama, "stream": False}
    ai_response = "Error en la respuesta del asistente"

    try:
        logger.info("Consultando a Ollama...")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        ai_response = data.get('response', 'No pude generar una respuesta.')
        logger.info("Respuesta de Ollama recibida")
    except requests.exceptions.RequestException as e:
        ai_response = f"Error de conexión con el asistente IA: {e}"
        logger.error(f"Error en Ollama: {e}")
    except Exception as e:
        ai_response = f"Error inesperado: {e}"
        logger.error(f"Error inesperado: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Consulta completada en {elapsed_time:.2f} segundos")
    
    return jsonify({"respuesta": ai_response, "tiempo": f"{elapsed_time:.2f} segundos", "pacientes_encontrados": len(found_patients)})


# --- Rutas de marcador de posición para agendar.html ---
@app.route('/verificar-rut')
def verificar_rut():
    return jsonify({'valido': True})

@app.route('/crear-reserva', methods=['POST'])
def crear_reserva():
    return jsonify(message='Reserva creada exitosamente!'), 201


if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)