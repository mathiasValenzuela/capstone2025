from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
from sklearn.cluster import KMeans
import json

app = Flask(__name__)

# Configuración MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['HOSPITAL']
pacientes = db['PACIENTES']

@app.route('/')
def index():
    pacientes_data = list(pacientes.find())
    return render_template('index.html', pacientes=pacientes_data)

@app.route('/dashboard')
def dashboard():
    # Procesamiento de datos
    prioridades = [float(p['NIVEL_PRIORIDAD'].replace('%', '')) for p in pacientes.find()]
    
    # Análisis con NumPy
    stats = {
        'max_priority': round(np.max(prioridades), 1),
        'avg_priority': round(np.mean(prioridades), 1),
        'min_priority': round(np.min(prioridades), 1)
    }
    
    # Machine Learning: Clustering
    X = np.array(prioridades).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3).fit(X)
    clusters = kmeans.labels_
    
    # Preparar datos para gráfico
    chart_data = {
        'labels': [p['NOMBRE_PACIENTE'] for p in pacientes.find()],
        'values': prioridades
    }
    
    return render_template('dashboard.html',
                         labels=json.dumps(chart_data['labels']),
                         values=json.dumps(chart_data['values']),
                         **stats)

if __name__ == '__main__':
    app.run(debug=True)