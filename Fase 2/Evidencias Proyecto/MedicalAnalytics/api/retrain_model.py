# -*- coding: utf-8 -*-
"""Script para reentrenar el modelo predictivo"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os
import logging
from sklearn.metrics import accuracy_score

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model():
    try:
        # 1. Cargar el dataset
        logger.info("Cargando dataset...")
        df = pd.read_csv('pacientes_01.csv')
        logger.info(f"Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
        
        # 2. Preprocesamiento de la variable objetivo
        df['Enfermedades_Lista'] = df['Enfermedades'].apply(
            lambda x: x.split(';') if isinstance(x, str) and ';' in x else [x] if x else [])
        
        df['Enfermedad_Principal'] = df['Enfermedades_Lista'].apply(
            lambda x: x[0].strip() if x else 'Ninguna')
        
        df_enfermos = df[df['Enfermedad_Principal'] != 'Ninguna'].copy()
        logger.info(f"Registros con enfermedades: {len(df_enfermos)}/{len(df)}")
        
        # 3. Preprocesamiento de características
        features = df_enfermos.drop(columns=[
            'PacienteID', 'RUT', 'Nombre', 'FechaNacimiento',
            'FechaAtencion', 'Enfermedades', 'Enfermedades_Lista'
        ])
        
        # 4. Codificación de variables
        categorical_features = ['Estado', 'Genero']
        numeric_features = features.select_dtypes(include=np.number).columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', StandardScaler(), numeric_features)
            ]
        )
        
        # 5. Preparar datos
        X = features.drop('Enfermedad_Principal', axis=1)
        y = df_enfermos['Enfermedad_Principal']
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # 6. Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # 7. Crear y entrenar modelo
        logger.info("Entrenando modelo...")
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        model.fit(X_train, y_train)
        logger.info("Entrenamiento completado")
        
        # 8. Evaluación
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Precisión del modelo: {accuracy:.4f}")
        
        # 9. Exportar modelo
        model_path = 'modelo_enfermedades_actualizado.pkl'
        joblib.dump({
            'model': model,
            'label_encoder': label_encoder,
            'feature_columns': X.columns.tolist(),
            'accuracy': accuracy
        }, model_path)
        
        logger.info(f"Modelo exportado exitosamente como '{model_path}'")
        return model_path, accuracy
        
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {str(e)}")
        return None, 0.0

if __name__ == '__main__':
    model_path, accuracy = train_model()
    if model_path:
        print(f"\n✅ Modelo reentrenado exitosamente: {model_path}")
        print(f"📊 Precisión del modelo: {accuracy:.2%}")
    else:
        print("\n❌ Error en el proceso de reentrenamiento")