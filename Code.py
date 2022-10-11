# Importing Libraries from matplotlib to visualize the data
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Importing Libraries to create GUI
from tkinter import *
import sqlite3

# Importing Libraries to perform calculations
import numpy as np
import pandas as pd
import os

# List of the symptoms is listed here in list l1.

l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
      'family_history', 'mucoid_sputum',
      'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
      'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
      'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
      'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
      'yellow_crust_ooze']

# List of Diseases is listed in list disease.

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
           ' Migraine', 'Cervical spondylosis',
           'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
           'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
           'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
           'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
           'Impetigo']

l2 = []
for i in range(0, len(l1)):
    l2.append(0)
#print(l2)

# Reading the training .csv file
DATA_PATH = "C:/Users/HP/OneDrive/Desktop/Training.csv"
df = pd.read_csv(DATA_PATH)

# Replace the values in the imported file by pandas by the inbuilt function replace in pandas.

df.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                          'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8,
                          'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12,
                          'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                          'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                          'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                          'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                          'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                          'Varicose veins': 30, 'Hypothyroidism': 31,
                          'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                          '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
                          'Psoriasis': 39,
                          'Impetigo': 40}}, inplace=True)

# printing the top 5 rows of the training dataset
df.head()


# Distribution graphs (histogram/bar graph) of column data



# Scatter and density plots


X = df[l1]
y = df[["prognosis"]]
np.ravel(y)
#print(X)
#print(y)

# Reading the  testing.csv file
tr = pd.read_csv("C:/Users/HP/OneDrive/Desktop/Testing.csv")

# Using inbuilt function replace in pandas for replacing the values

tr.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                          'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8,
                          'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12,
                          'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                          'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                          'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                          'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                          'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                          'Varicose veins': 30, 'Hypothyroidism': 31,
                          'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                          '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
                          'Psoriasis': 39,
                          'Impetigo': 40}}, inplace=True)

# printing the top 5 rows of the testing data
tr.head()


X_test = tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
#print(X_test)
#print(y_test)


# list1 = DF['prognosis'].unique()
root = Tk()
accuracy1=StringVar()
pred1=StringVar()
pred3 = StringVar()
def NaiveBayes():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp = messagebox.askokcancel("System", "Kindly Fill the Name")
        if comp:
            root.mainloop()

    elif ((Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here")):
        pred1.set(" ")
        sym = messagebox.askokcancel("System", "Kindly Fill atleast first two Symptoms")
        if sym:
            root.mainloop()

    else:
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb = gnb.fit(X, np.ravel(y))

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        y_pred = gnb.predict(X_test)
        #print("Naive Bayes")
        #print("Accuracy")
        ac=(accuracy_score(y_test, y_pred))
        ac=int(ac*100)
        #ac=accuracy_score(y_test, y_pred, normalize=False)
        #accuracy1.set(ac)
        #print("Confusion matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        #print(conf_matrix)

        psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
        for k in range(0, len(l1)):
            for z in psymptoms:
                if (z == l1[k]):
                    l2[k] = 1

        inputtest = [l2]
        predict = gnb.predict(inputtest)
        predicted = predict[0]

        h = 'no'
        for a in range(0, len(disease)):
            if (predicted == a):
                h = 'yes'
                break
        if (h == 'yes'):
            pred3.set(" ")
            pred3.set(disease[a])
        else:
            pred3.set(" ")
            pred3.set("Not Found")


        conn = sqlite3.connect('project.db')
        c = conn.cursor()
        c.execute(
            "CREATE TABLE IF NOT EXISTS Users(Name StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
        c.execute("INSERT INTO Users(Name,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?)",
                  (NameEn.get(), Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get(),
                   pred3.get()))
        conn.commit()
        c.close()
        conn.close()



# Tk class is used to create a root window
root.configure(background='Ivory')
root.title('Smart Disease Predictor System')
root.resizable(0, 0)

# taking first input as symptom
Symptom1 = StringVar()
Symptom1.set("Select Here")

# taking second input as symptom
Symptom2 = StringVar()
Symptom2.set("Select Here")

# taking third input as symptom
Symptom3 = StringVar()
Symptom3.set("Select Here")

# taking fourth input as symptom
Symptom4 = StringVar()
Symptom4.set("Select Here")

# taking fifth input as symptom
Symptom5 = StringVar()
Symptom5.set("Select Here")
Name = StringVar()

# function to Reset the given inputs to initial position
prev_win = None


NameLb = Label(root, text="Name of the Patient", fg="Red", bg="Ivory")
NameLb.config(font=("Times",15,"bold italic"))
NameLb.grid(row=6, column=0, pady=15, sticky=W)

def Reset():
    global prev_win

    Symptom1.set("Select Here")
    Symptom2.set("Select Here")
    Symptom3.set("Select Here")
    Symptom4.set("Select Here")
    Symptom5.set("Select Here")


    pred3.set(" ")

    try:
        prev_win.destroy()
        prev_win = None
    except AttributeError:
        pass


# Exit button to come out of system
from tkinter import messagebox


def Exit():
    qExit = messagebox.askyesno("System", "Do you want to exit the system")
    if qExit:
        root.destroy()
        exit()


# Headings for the GUI written at the top of GUI
w2 = Label(root, justify=LEFT, text="                SMART DISEASE PREDICTOR \n ", fg="Red", bg="Ivory")
w2.config(font=("Times", 30, "bold italic"))
w2.grid(row=1, column=0, columnspan=2, padx=100)

w2.config(font=("Times", 30, "bold italic"))
w2.grid(row=2, column=0, columnspan=2, padx=100)

# Label for the name

# Creating Labels for the symtoms
S1Lb = Label(root, text="Symptom 1", fg="Black", bg="Ivory")
S1Lb.config(font=("Times", 15, "bold italic"))
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="Black", bg="Ivory")
S2Lb.config(font=("Times", 15, "bold italic"))
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="Black", bg="Ivory")
S3Lb.config(font=("Times", 15, "bold italic"))
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="Black", bg="Ivory")
S4Lb.config(font=("Times", 15, "bold italic"))
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="Black", bg="Ivory")
S5Lb.config(font=("Times", 15, "bold italic"))
S5Lb.grid(row=11, column=0, pady=10, sticky=W)

# Labels for the different algorithms

Naiveb = Label(root, text="Prediction", fg="White", bg="green", width=20)
Naiveb.config(font=("Times", 15, "bold italic"))
Naiveb.grid(row=19, column=0, pady=10, sticky=W)

OPTIONS = sorted(l1)

# Taking name as input from user
NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

# Taking Symptoms as input from the dropdown from the user
S1 = OptionMenu(root, Symptom1, *OPTIONS)
S1.grid(row=7, column=1)

S2 = OptionMenu(root, Symptom2, *OPTIONS)
S2.grid(row=8, column=1)

S3 = OptionMenu(root, Symptom3, *OPTIONS)
S3.grid(row=9, column=1)

S4 = OptionMenu(root, Symptom4, *OPTIONS)
S4.grid(row=10, column=1)

S5 = OptionMenu(root, Symptom5, *OPTIONS)
S5.grid(row=11, column=1)

# Buttons for predicting the disease using different algorithms

lr = Button(root, text="PREDICT", command=NaiveBayes, bg="Blue", fg="white")
lr.config(font=("Times", 15, "bold italic"))
lr.grid(row=8, column=3, padx=10)



rs = Button(root, text="Reset Inputs", command=Reset, bg="yellow", fg="purple", width=15)
rs.config(font=("Times", 15, "bold italic"))
rs.grid(row=10, column=3, padx=10)

ex = Button(root, text="Exit System", command=Exit, bg="yellow", fg="purple", width=15)
ex.config(font=("Times", 15, "bold italic"))
ex.grid(row=11, column=3, padx=10)

# Showing the output of different algorithms


t3 = Label(root, font=("Times", 15, "bold italic"), text="Naive Bayes", height=1, bg="red"
           , width=40, fg="orange", textvariable=pred3, relief="sunken").grid(row=19, column=1, padx=10)


# calling this function because the application is ready to run
root.mainloop()

