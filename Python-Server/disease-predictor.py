from tkinter import *
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Symptom and disease lists
l1 = [
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
    "fluid_overload",
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
    " Migraine",
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
    "Heartattack",
    "Varicoseveins",
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

# Load and preprocess data
df = pd.read_csv("Training.csv")
df.replace({"prognosis": {d: i for i, d in enumerate(disease)}}, inplace=True)
X = df[l1]
y = df[["prognosis"]]

tr = pd.read_csv("Testing.csv")
tr.replace({"prognosis": {d: i for i, d in enumerate(disease)}}, inplace=True)
X_test = tr[l1]
y_test = tr[["prognosis"]]


# Prediction functions
def DecisionTree():
    clf3 = tree.DecisionTreeClassifier()
    clf3 = clf3.fit(X, y)

    y_pred = clf3.predict(X_test)
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
    print(
        "Decision Tree Correct Predictions:",
        accuracy_score(y_test, y_pred, normalize=False),
    )

    psymptoms = [
        Symptom1.get(),
        Symptom2.get(),
        Symptom3.get(),
        Symptom4.get(),
        Symptom5.get(),
    ]
    l2 = [0] * len(l1)  # Reset l2
    for k in range(len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted = predict[0]

    if predicted in range(len(disease)):
        t1.delete("1.0", END)
        t1.insert(END, disease[predicted])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")


def randomforest():
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X, np.ravel(y))

    y_pred = clf4.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print(
        "Random Forest Correct Predictions:",
        accuracy_score(y_test, y_pred, normalize=False),
    )

    psymptoms = [
        Symptom1.get(),
        Symptom2.get(),
        Symptom3.get(),
        Symptom4.get(),
        Symptom5.get(),
    ]
    l2 = [0] * len(l1)  # Reset l2
    for k in range(len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted = predict[0]

    if predicted in range(len(disease)):
        t2.delete("1.0", END)
        t2.insert(END, disease[predicted])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")


def NaiveBayes():
    gnb = GaussianNB()
    gnb = gnb.fit(X, np.ravel(y))

    y_pred = gnb.predict(X_test)
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
    print(
        "Naive Bayes Correct Predictions:",
        accuracy_score(y_test, y_pred, normalize=False),
    )

    psymptoms = [
        Symptom1.get(),
        Symptom2.get(),
        Symptom3.get(),
        Symptom4.get(),
        Symptom5.get(),
    ]
    l2 = [0] * len(l1)  # Reset l2
    for k in range(len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted = predict[0]

    if predicted in range(len(disease)):
        t3.delete("1.0", END)
        t3.insert(END, disease[predicted])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")


# GUI setup
root = Tk()
root.configure(background="blue")
root.title("Disease Predictor")

# Entry variables
Symptom1 = StringVar(value="")
Symptom2 = StringVar(value="")
Symptom3 = StringVar(value="")
Symptom4 = StringVar(value="")
Symptom5 = StringVar(value="")
Name = StringVar()

# Heading
w2 = Label(
    root,
    justify=LEFT,
    text="Disease Predictor using Machine Learning",
    fg="white",
    bg="blue",
)
w2.config(font=("Elephant", 30))
w2.grid(row=1, column=0, columnspan=2, padx=100)

w2 = Label(
    root, justify=LEFT, text="A Project by Yaswanth Sai Palaghat", fg="white", bg="blue"
)
w2.config(font=("Aharoni", 30))
w2.grid(row=2, column=0, columnspan=2, padx=100)

# Labels
NameLb = Label(root, text="Name of the Patient", fg="yellow", bg="black")
NameLb.grid(row=6, column=0, pady=15, sticky=W)

S1Lb = Label(root, text="Symptom 1", fg="yellow", bg="black")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="yellow", bg="black")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="yellow", bg="black")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="yellow", bg="black")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="yellow", bg="black")
S5Lb.grid(row=11, column=0, pady=10, sticky=W)

# Prediction labels (corrected)
dtLb = Label(root, text="Decision Tree", fg="white", bg="red")
dtLb.grid(row=15, column=0, pady=10, sticky=W)

rfLb = Label(root, text="Random Forest", fg="white", bg="red")
rfLb.grid(row=17, column=0, pady=10, sticky=W)

nbLb = Label(root, text="Naive Bayes", fg="white", bg="red")
nbLb.grid(row=19, column=0, pady=10, sticky=W)

# Entries
OPTIONS = sorted(l1)
NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

S1En = OptionMenu(root, Symptom1, *OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2, *OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3, *OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4, *OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5, *OPTIONS)
S5En.grid(row=11, column=1)

# Buttons
dst = Button(root, text="Decision Tree", command=DecisionTree, bg="green", fg="yellow")
dst.grid(row=8, column=3, padx=10)

rnf = Button(root, text="Random Forest", command=randomforest, bg="green", fg="yellow")
rnf.grid(row=9, column=3, padx=10)

lr = Button(root, text="Naive Bayes", command=NaiveBayes, bg="green", fg="yellow")
lr.grid(row=10, column=3, padx=10)

# Text fields
t1 = Text(root, height=1, width=40, bg="orange", fg="black")
t1.grid(row=15, column=1, padx=10)

t2 = Text(root, height=1, width=40, bg="orange", fg="black")
t2.grid(row=17, column=1, padx=10)

t3 = Text(root, height=1, width=40, bg="orange", fg="black")
t3.grid(row=19, column=1, padx=10)

root.mainloop()
