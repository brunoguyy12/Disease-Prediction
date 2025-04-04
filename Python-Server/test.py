from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Symptom and disease lists (keeping your original lists)
l1 = [...]  # Your symptom list
disease = [...]  # Your disease list

# Clean the disease list
disease = [d.strip() for d in disease]
mapping = {d: i for i, d in enumerate(disease)}

# Load and preprocess data
df = pd.read_csv("Training.csv")
df["prognosis"] = df["prognosis"].str.strip()
df.replace({"prognosis": mapping}, inplace=True)

X = df[l1]
y = df["prognosis"]

# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}

# Scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models and store accuracy metrics
model_metrics = {}
for name, model in models.items():
    # Use scaled data for SVM, original data for others
    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    accuracy = accuracy_score(y_test, y_pred)
    model_metrics[name] = {
        "accuracy": accuracy,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "model": model,
    }


# Ensemble prediction function
def ensemble_predict(symptoms):
    input_vector = np.zeros(len(l1))
    for symptom in symptoms:
        if symptom in l1:
            input_vector[l1.index(symptom)] = 1

    input_vector_scaled = scaler.transform([input_vector])
    predictions = {}

    for name, metrics in model_metrics.items():
        model = metrics["model"]
        if name == "SVM":
            pred = model.predict(input_vector_scaled)[0]
        else:
            pred = model.predict([input_vector])[0]
        predictions[name] = disease[pred]

    # Simple majority voting
    from collections import Counter

    votes = list(predictions.values())
    majority_prediction = Counter(votes).most_common(1)[0][0]

    return predictions, majority_prediction


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    model_predictions, ensemble_prediction = ensemble_predict(symptoms)

    return jsonify(
        {"predictions": model_predictions, "ensemble_prediction": ensemble_prediction}
    )


@app.route("/model_metrics", methods=["GET"])
def get_model_metrics():
    metrics = {
        name: {
            "accuracy": m["accuracy"],
            "cv_mean": m["cv_mean"],
            "cv_std": m["cv_std"],
        }
        for name, m in model_metrics.items()
    }
    return jsonify(metrics)


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({"symptoms": sorted(list(set(l1)))})


@app.route("/stats", methods=["GET"])
def stats():
    disease_counts = df["prognosis"].value_counts().to_dict()
    disease_names = {i: disease[i] for i in disease_counts.keys()}
    return jsonify({disease_names[k]: v for k, v in disease_counts.items()})


@app.route("/symptom_frequency", methods=["GET"])
def symptom_frequency():
    symptom_freq = df[l1].sum().to_dict()
    return jsonify(symptom_freq)


@app.route("/symptom_disease_relations", methods=["GET"])
def symptom_disease_relations():
    relations = {}
    for symptom in l1:
        diseases_with_symptom = df[df[symptom] == 1]["prognosis"].unique()
        diseases_names = [disease[i] for i in diseases_with_symptom]
        relations[symptom] = {"diseases": diseases_names, "count": len(diseases_names)}
    return jsonify(relations)


if __name__ == "__main__":
    # Print initial model performance
    for name, metrics in model_metrics.items():
        print(f"{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Cross-val mean: {metrics['cv_mean']:.4f}")
        print(f"  Cross-val std: {metrics['cv_std']:.4f}")

    app.run(host="0.0.0.0", port=5000, debug=True)
