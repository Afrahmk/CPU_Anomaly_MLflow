import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# ===========================
# 1. Simuler des données CPU
# ===========================
np.random.seed(42)
cpu_normal = np.random.normal(40, 5, 300)
cpu_spike = np.random.normal(90, 3, 10)
cpu_data = np.concatenate([cpu_normal, cpu_spike])
df = pd.DataFrame({"cpu": cpu_data})

# ===========================
# 2. Configurer MLflow avec SQLite
# ===========================
mlflow.set_tracking_uri("sqlite:///C:/Users/DELL/Desktop/pp/mlflow.db")
mlflow.set_experiment("CPU_Anomaly_Detection")  # Cette fois l'expérience doit apparaître

# ===========================
# 3. Lancer un run MLflow
# ===========================
with mlflow.start_run():

    n_estimators = 200
    contamination = 0.03
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("contamination", contamination)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42
    )
    model.fit(df)

    df["anomaly"] = model.predict(df)
    anomaly_count = (df["anomaly"] == -1).sum()
    anomaly_ratio = anomaly_count / len(df)
    mlflow.log_metric("anomalies_detected", anomaly_count)
    mlflow.log_metric("anomaly_ratio", anomaly_ratio)

    # Sauvegarder le modèle
    mlflow.sklearn.log_model(model, "isolation_forest_model")

    # Créer et sauvegarder le graphique
    plt.figure(figsize=(12,4))
    plt.plot(df['cpu'], label='CPU usage')
    plt.scatter(df.index[df['anomaly']==-1], df['cpu'][df['anomaly']==-1],
                color='red', label='Anomalies')
    plt.title("Détection d'anomalies CPU avec IsolationForest")
    plt.xlabel("Index")
    plt.ylabel("CPU Usage (%)")
    plt.legend()
    plt.savefig("cpu_anomalies.png")
    mlflow.log_artifact("cpu_anomalies.png")
    plt.close()

print("Terminé ! Lance 'mlflow ui --backend-store-uri sqlite:///C:/Users/DELL/Desktop/pp/mlflow.db'")
