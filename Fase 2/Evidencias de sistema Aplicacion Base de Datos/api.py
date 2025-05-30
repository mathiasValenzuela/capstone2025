from fastapi import FastAPI
from pymongo import MongoClient
from pymongo.cursor import CursorType
from threading import Thread
import time

app = FastAPI()

# Conexión a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["TEST1"]
examenes_col = db["EXAMENES"]

# Rangos normales de referencia (basados en literatura médica)
RANGOS_NORMALES = {
    "Hb (g/dL)": (1200, 1600),
    "Hto (%)": (3600, 4800),
    "Glóbulos Rojos (mill/µL)": (4200, 5800),
    "VCM (fL)": (8000, 10000),
    "CHCM (g/dL)": (3200, 3600),
    "Leucocitos (mil/µL)": (4000, 11000),
    "Plaquetas (mil/µL)": (150000, 450000),
    "VO2 máx (ml/kg/min)": (2500, 4000),
    "FC máx alcanzada (lpm)": (150, 200),
    "Presión sistólica post esfuerzo (mmHg)": (1400, 2000),
    "HRR (descenso en 1 min)": (120, 200),
    "Cloruro en sudor (mmol/L)": (100, 600),
    "Volumen de sudor (mg)": (5000, 10000),
    "Na⁺ (mmol/L)": (13500, 14500),
    "K⁺ (mmol/L)": (350, 510),
    "Glucosa en ayunas (mg/dL)": (7000, 10000),
    "Colesterol Total (mg/dL)": (12500, 20000),
    "HDL (mg/dL)": (4000, 6000),
    "LDL (mg/dL)": (7000, 13000),
    "Triglicéridos (mg/dL)": (5000, 15000),
    "IMC": (1850, 2499),
    "Presión Arterial Sistólica (mmHg)": (900, 1200),
    "Presión Arterial Diastólica (mmHg)": (600, 800)
}

def check_ranges(paciente: dict) -> dict:
    anomalias = []
    for key, valor in paciente.items():
        if key in RANGOS_NORMALES:
            min_val, max_val = RANGOS_NORMALES[key]
            if not (min_val <= valor <= max_val):
                anomalias.append({
                    "parametro": key,
                    "valor": valor,
                    "rango_optimo": f"{min_val}-{max_val}"
                })
    return {
        "paciente_id": paciente["Paciente"],
        "anomalias": anomalias,
        "sano": len(anomalias) == 0
    }

@app.get("/analizar-paciente/{paciente_id}")
async def analizar_paciente(paciente_id: str):
    paciente = examenes_col.find_one({"Paciente": paciente_id})
    if not paciente:
        return {"error": "Paciente no encontrado"}
    
    # Convertir valores numéricos
    for key in paciente.copy():
        if isinstance(paciente[key], str) and paciente[key].isdigit():
            paciente[key] = int(paciente[key])
    
    resultado = check_ranges(paciente)
    
    if not resultado["sano"]:
        print(f"\n⚠ ALERTA: Paciente {paciente_id} fuera de rangos óptimos")
        for anomalia in resultado["anomalias"]:
            print(f"   - {anomalia['parametro']}: {anomalia['valor']} (Rango óptimo: {anomalia['rango_optimo']})")
    
    return resultado

@app.get("/analizar-todos")
async def analizar_todos():
    pacientes_fuera_rango = []
    for paciente in examenes_col.find():
        resultado = check_ranges(paciente)
        if not resultado["sano"]:
            pacientes_fuera_rango.append(resultado)
            print(f"\n🔴 Paciente {resultado['paciente_id']} con {len(resultado['anomalias'])} anomalías")
    
    return {"total_pacientes": examenes_col.count_documents({}),
            "pacientes_fuera_rango": len(pacientes_fuera_rango),
            "detalles": pacientes_fuera_rango}

# Monitoreo en tiempo real
def monitor_en_tiempo_real():
    with examenes_col.watch(full_document='updateLookup') as stream:
        for change in stream:
            if change['operationType'] in ['insert', 'update']:
                paciente = change['fullDocument']
                resultado = check_ranges(paciente)
                if not resultado["sano"]:
                    print(f"\n🚨 ALERTA TIEMPO REAL - Paciente {paciente['Paciente']}")
                    for anomalia in resultado["anomalias"]:
                        print(f"   • {anomalia['parametro']}: {anomalia['valor']} vs {anomalia['rango_optimo']}")

# Iniciar monitoreo en segundo plano
Thread(target=monitor_en_tiempo_real, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)