#!/usr/bin/env python3

import argparse
import csv
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Use RandomForestRegressor for regression tasks
from sklearn.metrics import accuracy_score, classification_report  # Adjust metrics for regression if needed
from sklearn.preprocessing import StandardScaler  # Optional: Use for normalization if necessary

# # Step 4: Make predictions
# y_pred = model.predict(X_test)

# # Step 5: Evaluate the model
# # Classification metrics
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # For regression tasks, you can use metrics like Mean Squared Error (MSE)
# # from sklearn.metrics import mean_squared_error
# # print("MSE:", mean_squared_error(y_test, y_pred))

# # Step 6: (Optional) Save the model for future use
# import joblib
# joblib.dump(model, 'random_forest_model.pkl')

# # To load the model later:
# # model = joblib.load('random_forest_model.pkl')

# # Done!

def processPatientData(patientInfo):
    processed_patient = []

    #Add age
    processed_patient.append(patientInfo[0])

    #Add sex: one hot encoding
    if (patientInfo[1] == 'm'):
        processed_patient.append(0)
    else:
        processed_patient.append(1)

    #Get mean, std, median, range values from previous readings
    creatinineLevels = patientInfo[3:-2:2]
    creatinineLevels = [float(x) for x in creatinineLevels]
    # If no previous values, ifnore these metrics
    if (len(creatinineLevels) != 0):
        processed_patient.append(np.mean(creatinineLevels))
        processed_patient.append(np.std(creatinineLevels))
        processed_patient.append(np.median(creatinineLevels))
        processed_patient.append(np.ptp(creatinineLevels))
    else:
        processed_patient.append(np.nan)
        processed_patient.append(np.nan)
        processed_patient.append(np.nan)
        processed_patient.append(np.nan)

    #Add final creatinine level
    processed_patient.append(patientInfo[-1])

    return processed_patient

def preprocessData(data, labelsRow = None):
    
    # Extract features and labels
    features = data # Features
    train_labels = None
    # If lables are provided
    if (labelsRow):
        features = features.drop(data.columns[labelsRow], axis=1) # Features
        train_labels = data.iloc[:, labelsRow].values # Labels

    processed_features = []

    #Extract features
    for patientInfo in features.values:
        # Clip the end of the rows by removing NaN values
        isNan = pd.isna(patientInfo)
        cleaned_data = [x for x, nan in zip(patientInfo, isNan) if not nan]
        #Add patient to the featurs
        processed_features.append(processPatientData(cleaned_data))

    # Normalize features if needed
    scaler = StandardScaler()
    processed_features = scaler.fit_transform(processed_features)
    
    return processed_features, train_labels



def train(train_features, train_labels):

    # Initialize and train the Random Forest model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_features, train_labels)

    return model

def predict(data, model):
    '''
    Predict AKI from the datarow using the inputted model
    '''
    return model.predict(data)

def evaluate(test, model):
    return 0


def main():
    #inport
    data = pd.read_csv("training.csv")
    # Get features and labels
    train_features, train_labels = preprocessData(data, labelsRow=2)
    # Train the model
    model = train(train_features, train_labels)

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test.csv")
    parser.add_argument("--output", default="aki.csv")
    flags = parser.parse_args()
    r = csv.reader(open(flags.input))
    w = csv.writer(open(flags.output, "w"))
    w.writerow(("aki",))
    next(r) # skip headers
    for line in r:
        # TODO: Implement a better model
        w.writerow((random.choice(["y", "n"]),))

if __name__ == "__main__":
    main()