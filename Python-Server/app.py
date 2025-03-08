from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})


# Symptom and disease lists
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

# Clean the disease list
disease = [d.strip() for d in disease]
mapping = {d: i for i, d in enumerate(disease)}

# Load and train the model
df = pd.read_csv("Training.csv")


df["prognosis"] = df["prognosis"].str.strip()
df.replace({"prognosis": mapping}, inplace=True)

X = df[l1]
y = df["prognosis"]
model = RandomForestClassifier()
model.fit(X, np.ravel(y))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    input_vector = [0] * len(l1)
    for symptom in symptoms:
        if symptom in l1:
            input_vector[l1.index(symptom)] = 1

    prediction = model.predict([input_vector])[0]
    predicted_disease = disease[prediction]

    return jsonify({"disease": predicted_disease})


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({"symptoms": list(set(l1))})


@app.route("/", methods=["GET"])
def index():
    return "Hello, World!"


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
    app.run(host="0.0.0.0", port=5000, debug=True)
