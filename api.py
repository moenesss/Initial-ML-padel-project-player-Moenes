# ══════════════════════════════════════════════════════════════════════════════
# api.py — Padel Analytics · Players Module · Flask REST API
# ══════════════════════════════════════════════════════════════════════════════

import os
import logging
import traceback
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_logs.txt", mode="a"),
    ],
)
log = logging.getLogger(__name__)

app = Flask(__name__)

MLFLOW_EXPERIMENT_NAME = "Padel_Players_ML"
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_STORE = {
    "clf": None, "reg": None, "kmeans": None,
    "sc_clf": None, "sc_reg": None, "sc_cl": None,
    "df": None, "loaded_at": None, "run_id": None,
    "model_version": 0, "features_clf": None,
    "features_reg": None, "features_cl": None,
}

DROP_CLF = ["is_top_player", "ranking_position", "ranking_points",
            "performance_score", "cluster_kmeans", "cluster_hc"]
DROP_REG = ["contract_value_eur", "sponsorship_value_annual_eur",
            "is_top_player", "performance_score", "cluster_kmeans", "cluster_hc"]
CLUSTER_FEATURES = [
    "ranking_position", "total_titles", "win_rate_finals",
    "contract_value_eur", "instagram_followers_millions",
    "tiktok_followers_millions", "engagement_rate_percent",
    "total_social_followers", "sponsorship_value_annual_eur",
]


def train_models():
    log.info("Loading data from players_clean.csv ...")
    csv_path = os.path.join(os.path.dirname(__file__), "players_clean.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"players_clean.csv not found at {csv_path}.")

    df = pd.read_csv(csv_path)
    log.info(f"Data loaded: {df.shape[0]} players, {df.shape[1]} columns")
    df = df.fillna(0)
    log.info(f"NaN values filled with 0")

    for col in ["cluster_kmeans", "cluster_hc"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Fix class imbalance - ensure both classes exist for classifier
    if "is_top_player" in df.columns:
        class_counts = df["is_top_player"].value_counts()
        log.info(f"Class distribution: {class_counts.to_dict()}")
        if len(class_counts) < 2:
            log.warning("Only 1 class found — adding synthetic minority samples")
            minority_class = 1 if 0 in class_counts.index else 0
            n_synthetic = max(10, len(df) // 8)
            synthetic = df.sample(n=n_synthetic, replace=True, random_state=42).copy()
            synthetic["is_top_player"] = minority_class
            if minority_class == 1:
                synthetic["ranking_position"] = np.random.randint(1, 11, n_synthetic)
                synthetic["contract_value_eur"] = synthetic["contract_value_eur"] * 3
                synthetic["total_titles"] = synthetic["total_titles"] * 2 + 5
            else:
                synthetic["ranking_position"] = np.random.randint(50, 500, n_synthetic)
                synthetic["contract_value_eur"] = synthetic["contract_value_eur"] * 0.3
            df = pd.concat([df, synthetic], ignore_index=True)
            log.info(f"Added {n_synthetic} synthetic samples. New size: {len(df)}")

    MODEL_STORE["model_version"] += 1
    version = MODEL_STORE["model_version"]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"training_v{version}_{timestamp}") as run:
        run_id = run.info.run_id
        log.info(f"MLflow run started: {run_id}")

        mlflow.log_param("dataset_rows", df.shape[0])
        mlflow.log_param("dataset_cols", df.shape[1])
        mlflow.log_param("model_version", version)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("n_clusters", 3)
        mlflow.log_param("train_timestamp", timestamp)

        # CLASSIFIER
        drop_clf_actual = [c for c in DROP_CLF if c in df.columns]
        X_clf = df.drop(columns=drop_clf_actual)
        y_clf = df["is_top_player"]

        X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
            X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
        )

        sc_clf = StandardScaler()
        X_clf_train_sc = sc_clf.fit_transform(X_clf_train)
        X_clf_test_sc = sc_clf.transform(X_clf_test)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_clf_train_sc, y_clf_train)

        y_pred_clf = clf.predict(X_clf_test_sc)
        clf_accuracy = accuracy_score(y_clf_test, y_pred_clf)
        clf_f1 = f1_score(y_clf_test, y_pred_clf, zero_division=0)

        mlflow.log_metric("clf_accuracy", round(clf_accuracy, 4))
        mlflow.log_metric("clf_f1_score", round(clf_f1, 4))
        log.info(f"Classifier Accuracy: {clf_accuracy:.4f} | F1: {clf_f1:.4f}")

        sc_clf_full = StandardScaler()
        X_clf_full_sc = sc_clf_full.fit_transform(X_clf)
        clf.fit(X_clf_full_sc, y_clf)

        clf_path = os.path.join(MODELS_DIR, f"clf_v{version}.pkl")
        sc_clf_path = os.path.join(MODELS_DIR, f"sc_clf_v{version}.pkl")
        joblib.dump(clf, clf_path)
        joblib.dump(sc_clf_full, sc_clf_path)
        mlflow.log_artifact(clf_path, artifact_path="models")
        mlflow.log_artifact(sc_clf_path, artifact_path="models")

        # REGRESSOR
        drop_reg_actual = [c for c in DROP_REG if c in df.columns]
        X_reg = df.drop(columns=drop_reg_actual)
        y_reg = df["contract_value_eur"]

        X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )

        sc_reg = StandardScaler()
        X_reg_train_sc = sc_reg.fit_transform(X_reg_train)
        X_reg_test_sc = sc_reg.transform(X_reg_test)

        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_reg_train_sc, y_reg_train)

        y_pred_reg = reg.predict(X_reg_test_sc)
        reg_mae = mean_absolute_error(y_reg_test, y_pred_reg)
        reg_rmse = mean_squared_error(y_reg_test, y_pred_reg) ** 0.5
        reg_r2 = r2_score(y_reg_test, y_pred_reg)

        mlflow.log_metric("reg_mae", round(reg_mae, 2))
        mlflow.log_metric("reg_rmse", round(reg_rmse, 2))
        mlflow.log_metric("reg_r2", round(reg_r2, 4))
        log.info(f"Regressor MAE: {reg_mae:.0f} | RMSE: {reg_rmse:.0f} | R2: {reg_r2:.4f}")

        sc_reg_full = StandardScaler()
        X_reg_full_sc = sc_reg_full.fit_transform(X_reg)
        reg.fit(X_reg_full_sc, y_reg)

        reg_path = os.path.join(MODELS_DIR, f"reg_v{version}.pkl")
        sc_reg_path = os.path.join(MODELS_DIR, f"sc_reg_v{version}.pkl")
        joblib.dump(reg, reg_path)
        joblib.dump(sc_reg_full, sc_reg_path)
        mlflow.log_artifact(reg_path, artifact_path="models")
        mlflow.log_artifact(sc_reg_path, artifact_path="models")

        # CLUSTERING
        cl_features_actual = [c for c in CLUSTER_FEATURES if c in df.columns]
        X_cl = df[cl_features_actual].copy()

        sc_cl = StandardScaler()
        X_cl_sc = sc_cl.fit_transform(X_cl)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_cl_sc)

        labels = kmeans.labels_
        try:
            sil_score = silhouette_score(X_cl_sc, labels)
        except ValueError:
            sil_score = 0.0
            log.warning("Silhouette score could not be computed. Skipping.")

        mlflow.log_metric("kmeans_silhouette", round(float(sil_score), 4))
        log.info(f"KMeans Silhouette: {sil_score:.4f}")

        km_path = os.path.join(MODELS_DIR, f"kmeans_v{version}.pkl")
        sc_cl_path = os.path.join(MODELS_DIR, f"sc_cl_v{version}.pkl")
        joblib.dump(kmeans, km_path)
        joblib.dump(sc_cl, sc_cl_path)
        mlflow.log_artifact(km_path, artifact_path="models")
        mlflow.log_artifact(sc_cl_path, artifact_path="models")

        MODEL_STORE["clf"] = clf
        MODEL_STORE["reg"] = reg
        MODEL_STORE["kmeans"] = kmeans
        MODEL_STORE["sc_clf"] = sc_clf_full
        MODEL_STORE["sc_reg"] = sc_reg_full
        MODEL_STORE["sc_cl"] = sc_cl
        MODEL_STORE["df"] = df
        MODEL_STORE["loaded_at"] = datetime.utcnow().isoformat() + "Z"
        MODEL_STORE["run_id"] = run_id
        MODEL_STORE["features_clf"] = list(X_clf.columns)
        MODEL_STORE["features_reg"] = list(X_reg.columns)
        MODEL_STORE["features_cl"] = cl_features_actual

        log.info(f"All models ready. MLflow run: {run_id}")


def predict_one(row: pd.Series, player_index: int) -> dict:
    X_clf_row = row[MODEL_STORE["features_clf"]].values.reshape(1, -1)
    X_clf_scaled = MODEL_STORE["sc_clf"].transform(X_clf_row)
    is_top = int(MODEL_STORE["clf"].predict(X_clf_scaled)[0])

    # Safe probability extraction - handles both 1-class and 2-class
    proba = MODEL_STORE["clf"].predict_proba(X_clf_scaled)[0]
    classes = list(MODEL_STORE["clf"].classes_)
    if 1 in classes:
        top_prob = float(proba[classes.index(1)])
    else:
        top_prob = float(proba[0]) if is_top == 1 else 0.0

    X_reg_row = row[MODEL_STORE["features_reg"]].values.reshape(1, -1)
    X_reg_scaled = MODEL_STORE["sc_reg"].transform(X_reg_row)
    predicted_contract = float(MODEL_STORE["reg"].predict(X_reg_scaled)[0])

    X_cl_row = row[MODEL_STORE["features_cl"]].values.reshape(1, -1)
    X_cl_scaled = MODEL_STORE["sc_cl"].transform(X_cl_row)
    cluster = int(MODEL_STORE["kmeans"].predict(X_cl_scaled)[0])

    return {
        "player_index": player_index,
        "is_top_player": is_top,
        "top_player_probability": round(top_prob, 4),
        "predicted_contract_eur": round(predicted_contract, 2),
        "cluster": cluster,
        "model_version": MODEL_STORE["model_version"],
        "mlflow_run_id": MODEL_STORE["run_id"],
        "predicted_at": datetime.utcnow().isoformat() + "Z",
    }


@app.route("/health", methods=["GET"])
def health():
    if MODEL_STORE["clf"] is None:
        return jsonify({"status": "error", "models_loaded": False}), 503
    return jsonify({
        "status": "ok", "models_loaded": True,
        "loaded_at": MODEL_STORE["loaded_at"],
        "model_version": MODEL_STORE["model_version"],
        "mlflow_run_id": MODEL_STORE["run_id"],
        "player_count": len(MODEL_STORE["df"]),
        "features_clf": len(MODEL_STORE["features_clf"]),
        "features_reg": len(MODEL_STORE["features_reg"]),
    }), 200


@app.route("/players", methods=["GET"])
def get_players():
    if MODEL_STORE["df"] is None:
        return jsonify({"error": "Models not loaded"}), 503
    df = MODEL_STORE["df"].copy()
    top_only = request.args.get("top_only", "false").lower() == "true"
    limit = request.args.get("limit", None)
    if top_only and "is_top_player" in df.columns:
        df = df[df["is_top_player"] == 1]
    if limit:
        try:
            df = df.head(int(limit))
        except ValueError:
            pass
    players = df.replace({np.nan: None}).to_dict(orient="records")
    return jsonify({"count": len(players), "players": players}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL_STORE["clf"] is None:
        return jsonify({"error": "Models not loaded"}), 503
    try:
        body = request.get_json(force=True)
        if body is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400
        players_input = body.get("players", [body])
        predictions = []
        for i, player_data in enumerate(players_input):
            all_features = list(
                set(MODEL_STORE["features_clf"])
                | set(MODEL_STORE["features_reg"])
                | set(MODEL_STORE["features_cl"])
            )
            row = pd.Series({f: player_data.get(f, 0) for f in all_features})
            result = predict_one(row, player_index=i)
            if "player_id" in player_data:
                result["player_id"] = player_data["player_id"]
            predictions.append(result)
        log.info(f"POST /predict {len(predictions)} predictions returned")
        return jsonify({"predictions": predictions}), 200
    except Exception as e:
        log.error(f"Error in /predict: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500


@app.route("/predict/all", methods=["GET"])
def predict_all():
    if MODEL_STORE["clf"] is None:
        return jsonify({"error": "Models not loaded"}), 503
    try:
        df = MODEL_STORE["df"]
        predictions = []
        for i, (_, row) in enumerate(df.iterrows()):
            predictions.append(predict_one(row, player_index=i))
        log.info(f"GET /predict/all {len(predictions)} players processed")
        return jsonify({
            "run_at": datetime.utcnow().isoformat() + "Z",
            "player_count": len(predictions),
            "model_version": MODEL_STORE["model_version"],
            "mlflow_run_id": MODEL_STORE["run_id"],
            "predictions": predictions,
        }), 200
    except Exception as e:
        log.error(f"Error in /predict/all: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        log.info("Retraining triggered via POST /retrain ...")
        train_models()
        return jsonify({
            "status": "retrained",
            "loaded_at": MODEL_STORE["loaded_at"],
            "model_version": MODEL_STORE["model_version"],
            "mlflow_run_id": MODEL_STORE["run_id"],
        }), 200
    except Exception as e:
        log.error(f"Error in /retrain: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Retrain failed", "detail": str(e)}), 500


@app.route("/runs", methods=["GET"])
def list_runs():
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            return jsonify({"runs": [], "message": "No experiment found yet"}), 200
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )
        run_list = []
        for r in runs:
            run_list.append({
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "status": r.info.status,
                "start_time": datetime.utcfromtimestamp(r.info.start_time / 1000).isoformat() + "Z",
                "metrics": r.data.metrics,
                "params": r.data.params,
            })
        return jsonify({"experiment": MLFLOW_EXPERIMENT_NAME, "runs": run_list}), 200
    except Exception as e:
        log.error(f"Error in /runs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/update-dataset", methods=["POST"])
def update_dataset():
    try:
        body = request.get_json(force=True)
        csv_content = body.get("csv_content", "") if body else ""
        if not csv_content:
            return jsonify({"error": "No CSV content provided"}), 400
        csv_path = os.path.join(os.path.dirname(__file__), "players_clean.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)
        row_count = len(csv_content.strip().split("\n")) - 1
        log.info(f"Dataset updated: {row_count} players written to players_clean.csv")
        train_models()
        return jsonify({
            "status": "updated",
            "rows_written": row_count,
            "loaded_at": MODEL_STORE["loaded_at"],
            "model_version": MODEL_STORE["model_version"],
            "mlflow_run_id": MODEL_STORE["run_id"],
        }), 200
    except Exception as e:
        log.error(f"Error in /update-dataset: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Update failed", "detail": str(e)}), 500


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  Padel Analytics Players ML API  S12 MLflow Edition")
    log.info("=" * 60)
    try:
        train_models()
    except Exception as e:
        log.error(f"FATAL: Could not train models on startup: {e}")
        raise
    app.run(host="0.0.0.0", port=5000, debug=False)