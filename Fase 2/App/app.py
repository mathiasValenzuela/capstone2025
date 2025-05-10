from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
import numpy as np
import json
import re
import requests

app = Flask(__name__)

# Conexión MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['HOSPITAL']
pacientes = db['PACIENTES']

# Configuración Ollama
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

def formatear_contexto(paciente):
    return f"""
    👤 **Paciente**:
    - Nombre: {paciente.get('NOMBRE_PACIENTE', 'N/A')}
    - RUT: {paciente.get('RUT', 'N/A')}
    - Servicio: {paciente.get('SERVICIO_SOLICITANTE', 'N/A')}
    - Diagnóstico: {paciente.get('DIAGNOSTICO', 'No registrado')}
    - Último registro: {paciente.get('FECHA_INGRESO', 'Sin fecha')}
    """

def consultar_ia_chat(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": 4096,
            "stop": ["</s>", "[INST]"]
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=90)
        
        if response.status_code != 200:
            return f"🔴 Error Ollama ({response.status_code}): {response.text}"
            
        return response.json()['response'].strip()
        
    except requests.exceptions.ConnectionError:
        return "🔴 Error: Asegúrate que Ollama esté corriendo (ejecuta 'ollama serve')"
    except Exception as e:
        return f"🔴 Error: {str(e)}"

@app.route('/')
def index():
    pacientes_data = list(pacientes.find().sort("NIVEL_PRIORIDAD", -1))
    return render_template('index.html', pacientes=pacientes_data)

@app.route('/search')
def search():
    query = request.args.get('query', '')
    regex = re.compile(f'.*{re.escape(query)}.*', re.IGNORECASE)
    search_query = {
        '$or': [
            {'RUT': regex},
            {'NOMBRE_PACIENTE': regex},
            {'SERVICIO_SOLICITANTE': regex}
        ]
    }
    pacientes_data = list(pacientes.find(search_query))
    return render_template('index.html', pacientes=pacientes_data)

@app.route('/dashboard')
def dashboard():
    pipeline = [
        {
            '$project': {
                'SERVICIO_SOLICITANTE': 1,
                'NIVEL_PRIORIDAD': {
                    '$toDouble': {'$replaceOne': {'input': "$NIVEL_PRIORIDAD", 'find': "%", 'replacement': ""}}
                }
            }
        },
        {
            '$group': {
                '_id': "$SERVICIO_SOLICITANTE",
                'promedio_prioridad': {'$avg': "$NIVEL_PRIORIDAD"},
                'total_pacientes': {'$sum': 1}
            }
        },
        {'$sort': {'promedio_prioridad': -1}}
    ]

    servicios_data = list(pacientes.aggregate(pipeline))
    
    promedios = [s['promedio_prioridad'] for s in servicios_data]
    
    stats = {
        'max_priority': round(np.max(promedios), 1),
        'avg_priority': round(np.mean(promedios), 1),
        'min_priority': round(np.min(promedios), 1),
        'servicios': servicios_data
    }

    chart_data = {
        'labels': [s['_id'] for s in servicios_data],
        'values': [round(s['promedio_prioridad'], 1) for s in servicios_data]
    }

    return render_template('dashboard.html',
                         labels=json.dumps(chart_data['labels']),
                         values=json.dumps(chart_data['values']),
                         **stats)

@app.route("/chat", methods=["POST"])
def chat():
    mensaje = request.json.get("mensaje", "")
    
    contexto = ""
    if rut_match := re.search(r'\b(\d{1,2}\.?\d{3}\.?\d{3}-[\dkK])\b', mensaje, re.IGNORECASE):
        rut_limpio = rut_match.group(1).replace('.', '').upper()
        paciente = pacientes.find_one({"RUT": rut_limpio})
        contexto = formatear_contexto(paciente) if paciente else "❌ Paciente no encontrado"
    
    prompt = f"""<s>[INST] Eres un asistente médico especializado en análisis de datos hospitalarios. 
    Contexto del paciente:
    {contexto}
    
    Pregunta del usuario:
    {mensaje} [/INST]"""
    
    respuesta = consultar_ia_chat(prompt)
    return jsonify({"respuesta": respuesta})

if __name__ == '__main__':
    app.run(debug=True)