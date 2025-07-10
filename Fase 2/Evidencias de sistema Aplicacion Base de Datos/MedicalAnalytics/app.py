from flask import Flask, render_template, request, jsonify, redirect, url_for, g
from pymongo import MongoClient
import numpy as np
import json
import re
import requests
import pandas as pd
import joblib
from datetime import datetime, timedelta
from bson import ObjectId
from bson.errors import InvalidId
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from collections import Counter
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
try:
    import matplotlib
    matplotlib.use('Agg')  # Usar un backend que no requiera interfaz gráfica
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    print("Advertencia: matplotlib no está instalado. Algunas funciones de gráficos no estarán disponibles.")
from io import BytesIO
import base64
import math

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
db = client['MEDICALANALYTICS2']

# URL del servicio de predicción
PREDICTION_SERVICE_URL = 'http://localhost:5001/predict'

# Configuración para el modelo de lenguaje local (Ollama)
OLLAMA_API_URL = "http://localhost:11434/api/chat"  # URL CORREGIDA
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

# Columnas requeridas para el modelo predictivo
REQUIRED_COLUMNS = INDICADORES_CLAVE  # Usamos la misma lista

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
    logger.info(f"Iniciando cálculo de priorización para {len(pacientes_data)} pacientes.")
    if not pacientes_data:
        logger.info("No hay datos de pacientes para calcular priorización.")
        return []

    processed_pacientes = []
    # Primero, calculamos el coeficiente de error para todos los pacientes
    for p in pacientes_data:
        coeficiente_error = 0
        genero = p.get('Genero', 'hombre').lower()
        for key, ranges in HEALTH_RANGES.items():
            try:
                value_str = str(p.get(key, '0')).replace(',', '.')
                value = float(value_str)
            except (ValueError, TypeError):
                logger.warning(f"No se pudo convertir '{key}': '{p.get(key)}' a numérico para PacienteID: {p.get('PacienteID')}. Se omitirá en cálculo de error.")
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

    # Luego, encontramos el coeficiente máximo de los pacientes EN ESTE CONJUNTO DE DATOS (MES COMPLETO O RESULTADOS DE BÚSQUEDA)
    max_coeficiente = max((p.get('coeficiente_error', 0) for p in processed_pacientes), default=0)
    logger.info(f"Máximo coeficiente de error calculado: {max_coeficiente}")

    # Finalmente, calculamos la priorización basada en el max_coeficiente de este conjunto
    for p in processed_pacientes:
        p['priorizacion'] = round((p.get('coeficiente_error', 0) / max_coeficiente) * 100, 2) if max_coeficiente > 0 else 0
            
    logger.info("Cálculo de priorización completado.")
    return processed_pacientes

# NUEVA FUNCIÓN PARA VERIFICAR SALUD DEL SERVICIO
def verificar_salud_servicio_prediccion():
    """Verifica si el servicio de predicción está saludable y listo"""
    health_url = PREDICTION_SERVICE_URL.replace('/predict', '/health')
    try:
        response = requests.get(health_url, timeout=3)
        if response.status_code == 200:
            health_data = response.json()
            return health_data.get('model_loaded', False)
        return False
    except requests.exceptions.RequestException:
        return False

# Función que usa el servicio de predicción externo con procesamiento por lotes
def predecir_enfermedades_api(df, mes):
    logger.info(f"Enviando datos a API de predicción para el mes: {mes}. Filas: {len(df)}")
    
    # Verificar estado del servicio primero
    if not verificar_salud_servicio_prediccion():
        logger.error("🚨 Servicio de predicción NO está disponible o modelo no cargado")
        return None, df.copy()
    
    try:
        # Eliminamos columnas problemáticas
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])
        
        # Convertimos columnas de fecha a string
        date_columns = ['FechaNacimiento', 'FechaAtencion']
        for col in date_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Dividir en lotes para evitar timeout
        batch_size = 5000  # Tamaño del lote
        chunks = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
        
        all_predictions = []
        metrics = {
            'total_predicciones': 0,
            'distribucion_enfermedades': {},
            'model_accuracy': 0.8846
        }
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Procesando lote {i+1}/{len(chunks)} ({len(chunk)} registros)")
            
            # Convertir DataFrame a formato JSON
            data = chunk.to_dict(orient='records')
            response = requests.post(PREDICTION_SERVICE_URL, json=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                chunk_predictions = pd.DataFrame(result.get('predictions', []))
                all_predictions.append(chunk_predictions)
                
                # Acumular métricas
                if 'metrics' in result:
                    metrics['total_predicciones'] += result['metrics'].get('total_predicciones', 0)
                    
                    # Acumular distribución de enfermedades
                    for enfermedad, count in result['metrics'].get('distribucion_enfermedades', {}).items():
                        metrics['distribucion_enfermedades'][enfermedad] = metrics['distribucion_enfermedades'].get(enfermedad, 0) + count
                    
                    # Actualizar precisión del modelo
                    metrics['model_accuracy'] = result['metrics'].get('model_accuracy', 0.8846)
            else:
                logger.error(f"Error en API de predicción (lote {i+1}): {response.status_code} - {response.text}")
        
        # Combinar resultados
        if all_predictions:
            df_predicted = pd.concat(all_predictions, ignore_index=True)
            return metrics, df_predicted
        else:
            logger.warning("No se recibieron predicciones válidas")
            return None, df.copy()
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión con API de predicción: {str(e)}")
        return None, df.copy()
    except Exception as e:
        logger.error(f"Error inesperado al usar API de predicción: {str(e)}")
        return None, df.copy()

@app.route('/')
def index():
    mes = request.args.get('mes', 'ENERO')
    page = request.args.get('page', 1, type=int)
    per_page = 100 # Mostrar 100 pacientes por página

    coleccion = db[mes]
    
    # Obtener todos los pacientes para el mes para calcular la priorización y el coeficiente de error global
    all_pacientes_for_month = list(coleccion.find())
    
    # Calcular priorización para todos los pacientes del mes
    pacientes_con_priorizacion_full = calcular_priorizacion(all_pacientes_for_month)
    
    # MODIFICACIÓN CLAVE: Ordenar por coeficiente_error de forma descendente
    pacientes_con_priorizacion_full.sort(key=lambda x: x.get('coeficiente_error', 0), reverse=True)
    
    total_pacientes = len(pacientes_con_priorizacion_full)
    
    # Calcular el offset y obtener los pacientes para la página actual
    offset = (page - 1) * per_page
    pacientes_data_paginated = pacientes_con_priorizacion_full[offset:offset + per_page]
    
    # Calcular el número total de páginas
    total_pages = math.ceil(total_pacientes / per_page)

    return render_template('index.html', 
                           pacientes=pacientes_data_paginated, 
                           meses=MESES, 
                           mes_actual=mes,
                           total_pacientes=total_pacientes,
                           current_page=page,
                           per_page=per_page,
                           total_pages=total_pages)

@app.route('/search')
def search():
    query = request.args.get('query', '')
    mes = request.args.get('mes', 'ENERO')
    page = request.args.get('page', 1, type=int)
    per_page = 100 # Mantener la misma paginación para búsqueda

    coleccion = db[mes]
    regex = re.compile(f'.*{re.escape(query)}.*', re.IGNORECASE)
    
    # Definir las columnas en las que se realizará la búsqueda
    searchable_columns = ['Nombre', 'RUT', 'Estado', 'Genero', 'Edad', 'FechaNacimiento', 'FechaAtencion'] + INDICADORES_CLAVE
    
    # Construir la lista de condiciones OR para el filtro de búsqueda
    or_conditions = []
    for col in searchable_columns:
        or_conditions.append({col: regex})

    # Consulta de búsqueda
    search_filter = {'$or': or_conditions}
    
    # Obtener todos los resultados de la búsqueda para calcular la priorización y el coeficiente de error global
    all_matching_pacientes = list(coleccion.find(search_filter))
    
    # Calcular priorización para todos los pacientes que coinciden con la búsqueda
    pacientes_con_priorizacion_full_search = calcular_priorizacion(all_matching_pacientes)
    
    # MODIFICACIÓN CLAVE: Ordenar por coeficiente_error de forma descendente
    pacientes_con_priorizacion_full_search.sort(key=lambda x: x.get('coeficiente_error', 0), reverse=True)
    
    total_pacientes_busqueda = len(pacientes_con_priorizacion_full_search)
    
    # Calcular el offset y obtener los resultados de la búsqueda para la página actual
    offset = (page - 1) * per_page
    pacientes_data_paginated_search = pacientes_con_priorizacion_full_search[offset:offset + per_page]
    
    # Calcular el número total de páginas para los resultados de la búsqueda
    total_pages = math.ceil(total_pacientes_busqueda / per_page)

    return render_template('index.html', 
                           pacientes=pacientes_data_paginated_search, 
                           meses=MESES, 
                           mes_actual=mes,
                           total_pacientes=total_pacientes_busqueda,
                           current_page=page,
                           per_page=per_page,
                           total_pages=total_pages,
                           query=query)

@app.route('/dashboard')
def dashboard():
    mes = request.args.get('mes', 'ENERO')
    page = request.args.get('page', 1, type=int)
    per_page = 10

    coleccion = db[mes]
    total_pacientes = coleccion.count_documents({})
    logger.info(f"Dashboard cargado para el mes: {mes}. Total de pacientes en la colección: {total_pacientes}")
    
    # Obtener todos los pacientes para el mes para cálculos de dashboard
    pacientes_data_all = list(coleccion.find())
    logger.info(f"Pacientes cargados desde MongoDB: {len(pacientes_data_all)} registros.")
    
    # Calcular priorización y coeficiente_error para todos los pacientes
    pacientes_con_priorizacion_all = calcular_priorizacion(pacientes_data_all)
    
    # Convertir a DataFrame para predicción y asegurar tipos de datos
    df_all_patients = pd.DataFrame(pacientes_con_priorizacion_all)
    logger.info(f"DataFrame para predicción creado. Shape: {df_all_patients.shape}")

    # Asegurar columnas requeridas
    for col in REQUIRED_COLUMNS:
        if col not in df_all_patients.columns:
            df_all_patients[col] = 0.0
            logger.warning(f"Columna requerida '{col}' no encontrada en los datos. Se añadió con valor 0.")

    # Predicción de enfermedades usando el servicio externo
    datos_prediccion_enfermedades = {}
    df_predicted_enfermedades = df_all_patients.copy()
    
    logger.info("Usando servicio externo para predicción de enfermedades")
    metrics_enfermedades, df_predicted_enfermedades = predecir_enfermedades_api(df_all_patients.copy(), mes)
    
    if metrics_enfermedades:
        datos_prediccion_enfermedades = metrics_enfermedades
        logger.info("Predicción de enfermedades completada mediante servicio externo")
    else:
        logger.warning("No se pudo obtener predicción de enfermedades desde el servicio externo")
    
    # Pacientes críticos (con enfermedades predichas)
    critical_patients = []
    if 'enfermedad_predicha' in df_predicted_enfermedades.columns:
        critical_patients_df = df_predicted_enfermedades.copy()
        if not critical_patients_df.empty:
            critical_patients_df = critical_patients_df.sort_values(by='confianza_enfermedad', ascending=False)
            critical_patients = critical_patients_df.to_dict('records')
            logger.info(f"Pacientes críticos identificados: {len(critical_patients)}")
        else:
            logger.info("No se encontraron pacientes con enfermedades predichas en este mes.")
    else:
        logger.warning("La columna 'enfermedad_predicha' no existe en el DataFrame después de la predicción.")
    
    total_critical_patients = len(critical_patients)
    
    # CORRECCIÓN: Manejo adecuado de paginación cuando no hay pacientes
    if total_critical_patients > 0:
        total_critical_pages = math.ceil(total_critical_patients / per_page)
        offset = (page - 1) * per_page
        paginated_critical_patients = critical_patients[offset:offset + per_page]
    else:
        total_critical_pages = 0
        paginated_critical_patients = []
    
    logger.info(f"Paginación: Página {page}/{total_critical_pages}, mostrando {len(paginated_critical_patients)} pacientes críticos.")

    # Preparar datos para gráfico de enfermedades predichas
    enfermedades_predichas_labels = []
    enfermedades_predichas_values = []
    if datos_prediccion_enfermedades and 'distribucion_enfermedades' in datos_prediccion_enfermedades:
        # Ordenar por frecuencia descendente y tomar las 5 primeras
        distribucion = datos_prediccion_enfermedades['distribucion_enfermedades']
        sorted_distribucion = sorted(distribucion.items(), key=lambda x: x[1], reverse=True)[:5]
        enfermedades_predichas_labels = [item[0] for item in sorted_distribucion]
        enfermedades_predichas_values = [item[1] for item in sorted_distribucion]

    # --- Datos para Gráficos ---

    # Top 5 enfermedades reales (de la base de datos)
    pipeline_enfermedades = [
        {"$unwind": "$Enfermedades"},
        {"$match": {"Enfermedades": {"$nin": ["Ninguna", "ninguna", "Sano", "sano", "", None]}}},
        {"$group": {"_id": "$Enfermedades", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5}
    ]
    top_enfermedades_data = list(coleccion.aggregate(pipeline_enfermedades))
    top_enfermedades_labels = [e['_id'] for e in top_enfermedades_data]
    top_enfermedades_values = [e['count'] for e in top_enfermedades_data]
    logger.info(f"Datos Top 5 Enfermedades: Labels={top_enfermedades_labels}, Values={top_enfermedades_values}")

    # Priorización de pacientes (para gráfico de burbujas y barras)
    pacientes_con_priorizacion = calcular_priorizacion(pacientes_data_all)
    pacientes_con_priorizacion.sort(key=lambda x: x.get('priorizacion', 0), reverse=True)
    
    # Datos para gráfico de barras de priorización (Top 10)
    priorizacion_labels = [p.get('Nombre', 'N/A') for p in pacientes_con_priorizacion[:10]]
    priorizacion_values = [p.get('priorizacion', 0) for p in pacientes_con_priorizacion[:10]]
    logger.info(f"Datos Priorización Top 10 (Gráfico de Barras): Labels={priorizacion_labels}, Values={priorizacion_values}")

    # Estado de salud
    pipeline_estado = [{"$group": {"_id": "$Estado", "count": {"$sum": 1}}}]
    estado_data = list(coleccion.aggregate(pipeline_estado))
    estado_labels = [e.get('_id', 'N/A') for e in estado_data]
    estado_values = [e.get('count', 0) for e in estado_data]
    logger.info(f"Datos Estado de Salud (Gráfico de Dona): Labels={estado_labels}, Values={estado_values}")
    
    # Distribución por Género
    pipeline_genero = [{"$group": {"_id": "$Genero", "count": {"$sum": 1}}}]
    genero_data = list(coleccion.aggregate(pipeline_genero))
    genero_labels = [g.get('_id', 'N/A') for g in genero_data]
    genero_values = [g.get('count', 0) for g in genero_data]
    logger.info(f"Datos Distribución por Género (Gráfico de Pastel): Labels={genero_labels}, Values={genero_values}")
    
    # Datos para gráfico de burbujas (Priorización vs Edad vs IMC)
    bubble_chart_data = []
    for p in pacientes_con_priorizacion:
        try:
            edad = float(p.get('Edad'))
            priorizacion = float(p.get('priorizacion'))
            imc = float(p.get('IMC', 0))
            bubble_chart_data.append({'x': edad, 'y': priorizacion, 'r': max(imc / 2, 5)})
        except (TypeError, ValueError) as e:
            logger.warning(f"Error al procesar datos para gráfico de burbujas para PacienteID {p.get('PacienteID')}: {e}")
            continue
    logger.info(f"Datos Gráfico de Burbujas generados: {len(bubble_chart_data)} puntos.")

    # Datos para gráfico de dispersión (Glucosa vs Edad)
    scatter_glucosa_data = []
    for p in pacientes_data_all:
        try:
            edad = float(p.get('Edad'))
            glucosa_str = str(p.get('Glucosa_en_ayunas_mg/dL', '0')).replace(',', '.')
            glucosa = float(glucosa_str)
            scatter_glucosa_data.append({'x': edad, 'y': glucosa})
        except (TypeError, ValueError) as e:
            logger.warning(f"Error al procesar datos para gráfico Glucosa vs Edad para PacienteID {p.get('PacienteID')}: {e}")
            continue
    logger.info(f"Datos Glucosa vs Edad generados: {len(scatter_glucosa_data)} puntos.")

    # Datos para gráfico de dispersión (Colesterol Total vs Edad)
    scatter_colesterol_data = []
    for p in pacientes_data_all:
        try:
            edad = float(p.get('Edad'))
            colesterol_str = str(p.get('Colesterol_Total_mg/dL', '0')).replace(',', '.')
            colesterol = float(colesterol_str)
            scatter_colesterol_data.append({'x': edad, 'y': colesterol})
        except (TypeError, ValueError) as e:
            logger.warning(f"Error al procesar datos para gráfico Colesterol vs Edad para PacienteID {p.get('PacienteID')}: {e}")
            continue
    logger.info(f"Datos Colesterol Total vs Edad generados: {len(scatter_colesterol_data)} puntos.")

    # Datos para gráfico de dispersión (Presión Sistólica vs Edad)
    scatter_presion_sistolica_data = []
    for p in pacientes_data_all:
        try:
            edad = float(p.get('Edad'))
            presion_str = str(p.get('Presión_Arterial_Sistólica_mmHg', '0')).replace(',', '.')
            presion = float(presion_str)
            scatter_presion_sistolica_data.append({'x': edad, 'y': presion})
        except (TypeError, ValueError) as e:
            logger.warning(f"Error al procesar datos para gráfico Presión Sistólica vs Edad para PacienteID {p.get('PacienteID')}: {e}")
            continue
    logger.info(f"Datos Presión Sistólica vs Edad generados: {len(scatter_presion_sistolica_data)} puntos.")

    modelos_recomendados = [
        {"nombre": "Random Forest", "precision": 88.46, "datos_procesados": "500K registros", "ventajas": "Robusto contra overfitting, manejo de características no lineales"},
        {"nombre": "XGBoost", "precision": 92, "datos_procesados": ">1M registros", "ventajas": "Alto rendimiento con grandes datasets"},
        {"nombre": "Redes Neuronales", "precision": 91, "datos_procesados": "2M registros", "ventajas": "Captura relaciones complejas"},
        {"nombre": "LightGBM", "precision": 93, "datos_procesados": "1.5M registros", "ventajas": "Entrenamiento rápido"}
    ]

    # Obtener la precisión del modelo desde el servicio si está disponible
    model_accuracy_percent = 88.46  # Valor por defecto en porcentaje
    if metrics_enfermedades and 'model_accuracy' in metrics_enfermedades:
        # Convertir a porcentaje: el servicio devuelve 0.8846, lo multiplicamos por 100 para mostrar
        model_accuracy_percent = round(metrics_enfermedades['model_accuracy'] * 100, 2)
    elif datos_prediccion_enfermedades and 'model_accuracy' in datos_prediccion_enfermedades:
        model_accuracy_percent = round(datos_prediccion_enfermedades['model_accuracy'] * 100, 2)

    return render_template('dashboard.html', 
        total_pacientes=total_pacientes, 
        top_enfermedades_labels=json.dumps(top_enfermedades_labels), 
        top_enfermedades_values=json.dumps(top_enfermedades_values), 
        meses=MESES, 
        mes_actual=mes, 
        prediccion_enfermedades=datos_prediccion_enfermedades,
        enfermedades_predichas_labels=json.dumps(enfermedades_predichas_labels),
        enfermedades_predichas_values=json.dumps(enfermedades_predichas_values),
        priorizacion_labels=json.dumps(priorizacion_labels),
        priorizacion_values=json.dumps(priorizacion_values),
        estado_labels=json.dumps(estado_labels),
        estado_values=json.dumps(estado_values),
        genero_labels=json.dumps(genero_labels),
        genero_values=json.dumps(genero_values),
        bubble_chart_data=json.dumps(bubble_chart_data),
        scatter_glucosa_data=json.dumps(scatter_glucosa_data),
        scatter_colesterol_data=json.dumps(scatter_colesterol_data),
        scatter_presion_sistolica_data=json.dumps(scatter_presion_sistolica_data),
        modelos_recomendados=modelos_recomendados,
        critical_patients=paginated_critical_patients,
        total_critical_patients=total_critical_patients,
        current_critical_page=page,
        total_critical_pages=total_critical_pages,
        per_page=per_page,
        model_accuracy=model_accuracy_percent  # Porcentaje para mostrar
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
                        if key in INDICADORES_CLAVE:
                            datos_actualizados[key] = float(str(value).replace(',', '.'))
                        else:
                            datos_actualizados[key] = value
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
            # MEJORA: Manejo robusto de RUT con caracteres especiales
            rut = re.sub(r'[^0-9kK]', '', query).upper()
            # Formatear RUT con guión si es necesario
            if len(rut) > 1:
                rut = rut[:-1] + '-' + rut[-1]
            regex = re.compile(f'^{re.escape(rut)}', re.IGNORECASE)
            pacientes = list(coleccion.find({'RUT': regex}))
        elif search_type == 'Nombre':
            # MEJORA: Búsqueda flexible de nombres
            regex = re.compile(f'.*{re.escape(query)}.*', re.IGNORECASE)
            pacientes = list(coleccion.find({'Nombre': regex}))
        else:
            # MEJORA: Búsqueda en múltiples campos
            regex = re.compile(f'.*{re.escape(query)}.*', re.IGNORECASE)
            searchable_cols = ['Nombre', 'RUT', 'Enfermedades']
            or_conditions = [ {col: regex} for col in searchable_cols ]
            pacientes = list(coleccion.find({'$or': or_conditions}))
        
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
    
    # MEJORA: Expresiones regulares más flexibles
    rut_match = re.search(r'[\d\.]{7,10}-?[\dkK]', user_message, re.IGNORECASE)
    name_match = re.search(r'(paciente|nombre|name|buscar|información|datos|historial)[\s:]*([\w\sáéíóúÁÉÍÓÚñÑ]+)', 
                          user_message, re.IGNORECASE)
    
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
        search_query = rut_match.group(0).strip()
        search_type = "RUT"
        logger.info(f"Búsqueda por RUT detectada: {search_query}")
    elif name_match and name_match.group(2):
        search_query = name_match.group(2).strip()
        search_type = "Nombre"
        logger.info(f"Búsqueda por nombre detectada: {search_query}")
    else:
        search_query = user_message
        search_type = "General"
        logger.warning("No se detectó RUT ni nombre explícito, realizando búsqueda general.")
    
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
            # Iterar sobre cada campo del documento del paciente
            for key, value in paciente.items():
                # Excluir campos internos de MongoDB si no son relevantes para la IA
                if key not in ['_id']:
                    if isinstance(value, list):
                        patient_info_context += f"- {key}: {', '.join(map(str, value))}\n"
                    elif isinstance(value, datetime):
                        patient_info_context += f"- {key}: {value.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    else:
                        patient_info_context += f"- {key}: {value}\n"
            patient_info_context += "\n"
    else:
        patient_info_context = "⚠️ No se encontraron pacientes con los datos proporcionados."
        logger.warning("No se encontraron pacientes")
    
    prompt_for_ollama = (
        "Eres un asistente médico experto. El usuario ha solicitado información sobre pacientes. "
        f"Contexto de la base de datos:\n{patient_info_context}\n\n"
        f"Pregunta del usuario: {user_message}\n\n"
        "Responde de manera completa y profesional, utilizando toda la información relevante proporcionada en el contexto. "
        "Si el usuario pregunta sobre términos médicos, explica de forma simple pero precisa. "
        "Para datos numéricos, indica si están dentro de rangos saludables si tienes la información de rangos, o si son valores que requieren atención."
    )

    # CORRECCIÓN: Usar API de chat correcta
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt_for_ollama}
        ],
        "stream": False
    }

    ai_response = "Error en la respuesta del asistente"

    try:
        logger.info("Consultando a Ollama...")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        ai_response = data.get('message', {}).get('content', 'No pude generar una respuesta.')
        logger.info("Respuesta de Ollama recibida")
    except requests.exceptions.RequestException as e:
        ai_response = f"Error de conexión con el asistente IA: {e}"
        logger.error(f"Error en Ollama: {e}")
    except Exception as e:
        ai_response = f"Error inesperado: {e}"
        logger.error(f"Error inesperado: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Consulta completada en {elapsed_time:.2f} segundos")
    
    return jsonify({
        "respuesta": ai_response, 
        "tiempo": f"{elapsed_time:.2f} segundos", 
        "pacientes_encontrados": len(found_patients)
    })

@app.route('/verificar-rut')
def verificar_rut():
    return jsonify({'valido': True})

@app.route('/crear-reserva', methods=['POST'])
def crear_reserva():
    try:
        reserva_data = request.json
        # Opcional: Validar y limpiar los datos de reserva antes de insertar
        required_fields = ['rut', 'nombre', 'fecha', 'hora', 'salaId']
        if not all(field in reserva_data and reserva_data[field] for field in required_fields):
            logger.error(f"Datos de reserva incompletos: {reserva_data}")
            return jsonify(error='Faltan campos obligatorios para la reserva'), 400

        # Convertir fecha y hora a objetos datetime si es necesario para búsquedas o ordenamiento
        # En este caso, para simplemente guardar y mostrar, string está bien.

        db.RESERVAS.insert_one(reserva_data)
        logger.info(f"Reserva creada exitosamente: {reserva_data}")
        return jsonify(message='Reserva creada exitosamente!'), 201
    except Exception as e:
        logger.error(f"Error al crear reserva: {str(e)}")
        return jsonify(error=f"Error interno al crear la reserva: {str(e)}"), 500

@app.route('/obtener-reservas', methods=['GET'])
def obtener_reservas():
    try:
        reservas = list(db.RESERVAS.find({}, {'_id': 0})) # Excluir _id al enviar a JSON
        logger.info(f"Se recuperaron {len(reservas)} reservas.")
        return jsonify(reservas), 200
    except Exception as e:
        logger.error(f"Error al obtener reservas: {str(e)}")
        return jsonify(error=f"Error interno al obtener las reservas: {str(e)}"), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)