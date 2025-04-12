from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Symptom and disease lists (unchanged)
l1 = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions",
    "continuous_sneezing",
    "shivering",
    "chills",
    "joint_pain",
    "stomach_pain",
    "acidity",
    "ulcers_on_tongue",
    "muscle_wasting",
    "vomiting",
    "burning_micturition",
    "spotting_ urination",
    "fatigue",
    "weight_gain",
    "anxiety",
    "cold_hands_and_feets",
    "mood_swings",
    "weight_loss",
    "restlessness",
    "lethargy",
    "patches_in_throat",
    "irregular_sugar_level",
    "cough",
    "high_fever",
    "sunken_eyes",
    "breathlessness",
    "sweating",
    "dehydration",
    "indigestion",
    "headache",
    "yellowish_skin",
    "dark_urine",
    "nausea",
    "loss_of_appetite",
    "pain_behind_the_eyes",
    "back_pain",
    "constipation",
    "abdominal_pain",
    "diarrhoea",
    "mild_fever",
    "yellow_urine",
    "yellowing_of_eyes",
    "acute_liver_failure",
    "fluid_overload",
    "swelling_of_stomach",
    "swelled_lymph_nodes",
    "malaise",
    "blurred_and_distorted_vision",
    "phlegm",
    "throat_irritation",
    "redness_of_eyes",
    "sinus_pressure",
    "runny_nose",
    "congestion",
    "chest_pain",
    "weakness_in_limbs",
    "fast_heart_rate",
    "pain_during_bowel_movements",
    "pain_in_anal_region",
    "bloody_stool",
    "irritation_in_anus",
    "neck_pain",
    "dizziness",
    "cramps",
    "bruising",
    "obesity",
    "swollen_legs",
    "swollen_blood_vessels",
    "puffy_face_and_eyes",
    "enlarged_thyroid",
    "brittle_nails",
    "swollen_extremeties",
    "excessive_hunger",
    "extra_marital_contacts",
    "drying_and_tingling_lips",
    "slurred_speech",
    "knee_pain",
    "hip_joint_pain",
    "muscle_weakness",
    "stiff_neck",
    "swelling_joints",
    "movement_stiffness",
    "spinning_movements",
    "loss_of_balance",
    "unsteadiness",
    "weakness_of_one_body_side",
    "loss_of_smell",
    "bladder_discomfort",
    "foul_smell_of urine",
    "continuous_feel_of_urine",
    "passage_of_gases",
    "internal_itching",
    "toxic_look_(typhos)",
    "depression",
    "irritability",
    "muscle_pain",
    "altered_sensorium",
    "red_spots_over_body",
    "belly_pain",
    "abnormal_menstruation",
    "dischromic _patches",
    "watering_from_eyes",
    "increased_appetite",
    "polyuria",
    "family_history",
    "mucoid_sputum",
    "rusty_sputum",
    "lack_of_concentration",
    "visual_disturbances",
    "receiving_blood_transfusion",
    "receiving_unsterile_injections",
    "coma",
    "stomach_bleeding",
    "distention_of_abdomen",
    "history_of_alcohol_consumption",
    "fluid_overload.1",
    "blood_in_sputum",
    "prominent_veins_on_calf",
    "palpitations",
    "painful_walking",
    "pus_filled_pimples",
    "blackheads",
    "scurring",
    "skin_peeling",
    "silver_like_dusting",
    "small_dents_in_nails",
    "inflammatory_nails",
    "blister",
    "red_sore_around_nose",
    "yellow_crust_ooze",
]

disease = [
    "Fungal infection",
    "Allergy",
    "GERD",
    "Chronic cholestasis",
    "Drug Reaction",
    "Peptic ulcer diseae",
    "AIDS",
    "Diabetes",
    "Gastroenteritis",
    "Bronchial Asthma",
    "Hypertension",
    "Migraine",
    "Cervical spondylosis",
    "Paralysis (brain hemorrhage)",
    "Jaundice",
    "Malaria",
    "Chicken pox",
    "Dengue",
    "Typhoid",
    "hepatitis A",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Hepatitis E",
    "Alcoholic hepatitis",
    "Tuberculosis",
    "Common Cold",
    "Pneumonia",
    "Dimorphic hemmorhoids(piles)",
    "Heart attack",
    "Varicose veins",
    "Hypothyroidism",
    "Hyperthyroidism",
    "Hypoglycemia",
    "Osteoarthristis",
    "Arthritis",
    "(vertigo) Paroymsal  Positional Vertigo",
    "Acne",
    "Urinary tract infection",
    "Psoriasis",
    "Impetigo",
]

# Clean disease list and create mapping
disease = [d.strip() for d in disease]
mapping = {d: i for i, d in enumerate(disease)}

# Load and preprocess Training data
df_train = pd.read_csv("Training.csv")
df_train["prognosis"] = df_train["prognosis"].str.strip()
df_train.replace({"prognosis": mapping}, inplace=True)
X_train = df_train[l1]
y_train = df_train["prognosis"]

# Load and preprocess Testing data
df_test = pd.read_csv("Testing.csv")
df_test["prognosis"] = df_test["prognosis"].str.strip()
df_test.replace({"prognosis": mapping}, inplace=True)
X_test = df_test[l1]
y_test = df_test["prognosis"]

# Train individual models on Training.csv
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
lr_model = LogisticRegression(max_iter=1000, random_state=42)

rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Evaluate models on Testing.csv
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
svm_acc = accuracy_score(y_test, svm_model.predict(X_test))
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

print(f"RandomForest Accuracy: {rf_acc:.4f}")
print(f"SVM Accuracy: {svm_acc:.4f}")
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")


# Train ensemble model on Training.csv
ensemble_model = VotingClassifier(
    estimators=[("rf", rf_model), ("svm", svm_model), ("lr", lr_model)], voting="soft"
)
ensemble_model.fit(X_train, y_train)
ensemble_acc = accuracy_score(y_test, ensemble_model.predict(X_test))
print(f"\n\nEnsemble Accuracy: {ensemble_acc:.4f}\n\n")

# Generate predictions from the ensemble model
ensemble_preds = ensemble_model.predict(X_test)

# Print classification report
report = classification_report(y_test, ensemble_preds, target_names=disease)
print("\nClassification Report for Final Ensemble Model:\n")
print(report)

# Bar chart of model accuracies (Figure 5.1)
model_names = ["Random Forest", "SVM", "Logistic Regression", "Ensemble"]
accuracies = [rf_acc * 100, svm_acc * 100, lr_acc * 100, ensemble_acc * 100]
colors = ["skyblue", "lightgreen", "salmon", "gold"]

plt.figure(figsize=(8, 6))
sns.barplot(x=model_names, y=accuracies, palette=colors)

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc:.2f}%", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")  # Save it as a file for documentation
plt.show()


cf_matrix = confusion_matrix(y_test, ensemble_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Ensemble Model on Test Data")
plt.show()

# # Convert rows to tuples to make them hashable for set operations
# train_rows = set(tuple(row) for row in df_train[l1].to_numpy())
# test_rows = set(tuple(row) for row in df_test[l1].to_numpy())

# # Find the intersection (common rows)
# overlap = train_rows.intersection(test_rows)
# print(
#     f"Number of overlapping rows between Training and Testing sets based on symptoms: {len(overlap)}"
# )

# print(f"Testing.csv size: {len(df_test)}")


# Flask routes (predict endpoint updated to use ensemble_model)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    input_vector = [0] * len(l1)
    for symptom in symptoms:
        if symptom in l1:
            input_vector[l1.index(symptom)] = 1

    prediction = ensemble_model.predict([input_vector])[0]
    predicted_disease = disease[prediction]

    # Get the probabilities for all diseases
    probs = ensemble_model.predict_proba([input_vector])[0]

    # Find the top 3 diseases and their probabilities
    top_3_indices = np.argsort(probs)[-3:][
        ::-1
    ]  # Get indices of top 3 in descending order
    top_3_diseases = [disease[i] for i in top_3_indices]
    top_3_probs = [probs[i] * 100 for i in top_3_indices]  # Convert to percentages

    # Format the top 3 as a list of strings (e.g., "Diabetes: 24.00%")
    top_3_formatted = [
        f"{disease}: {prob:.2f}%" for disease, prob in zip(top_3_diseases, top_3_probs)
    ]
    print(top_3_formatted)

    return jsonify({"disease": predicted_disease, "probabilities": top_3_formatted})


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({"symptoms": list(set(l1)), "accuracy": f"{ensemble_acc:.4f}"})


@app.route("/", methods=["GET"])
def index():
    return "Hello, World!"


@app.route("/stats", methods=["GET"])
def stats():
    disease_counts = df_train["prognosis"].value_counts().to_dict()
    disease_names = {i: disease[i] for i in disease_counts.keys()}
    return jsonify({disease_names[k]: v for k, v in disease_counts.items()})


@app.route("/symptom_frequency", methods=["GET"])
def symptom_frequency():
    symptom_freq = df_train[l1].sum().to_dict()
    return jsonify(symptom_freq)


@app.route("/symptom_disease_relations", methods=["GET"])
def symptom_disease_relations():
    relations = {}
    for symptom in l1:
        diseases_with_symptom = df_train[df_train[symptom] == 1]["prognosis"].unique()
        diseases_names = [disease[i] for i in diseases_with_symptom]
        relations[symptom] = {"diseases": diseases_names, "count": len(diseases_names)}
    return jsonify(relations)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
