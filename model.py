#!/usr/bin/env python3

import argparse
import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import StandardScaler

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
    labels = None
    # If lables are provided
    if (labelsRow):
        features = features.drop(data.columns[labelsRow], axis=1) # Features
        labels = data.iloc[:, labelsRow].values # Labels

    processed_features = []

    #Extract features
    for patientInfo in features.values:
        # Clip the end of the rows by removing NaN values
        isNan = pd.isna(patientInfo)
        Nan_removed_data = [x for x, nan in zip(patientInfo, isNan) if not nan]
        #Add patient to the featurs
        processed_features.append(processPatientData(Nan_removed_data))

    # Normalize features
    scaler = StandardScaler()
    processed_features = scaler.fit_transform(processed_features)
    
    return processed_features, labels

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

def evaluate(real_labels, predicted_labels):
    f3_score = fbeta_score(real_labels, predicted_labels, beta=3, average='weighted')
    return f3_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test.csv")
    parser.add_argument("--output", default="aki.csv")
    flags = parser.parse_args()
    r = csv.reader(open(flags.input))
    headers = next(r)
    #Add to array to convert to pandas df
    testData = []
    for line in r:
        testData.append(line)
    #Convert to pandas df
    testData = pd.DataFrame(testData, columns=headers)
    testData.replace('', np.nan, inplace=True)

    #Train the model

    #import training data
    trainingData = pd.read_csv("training.csv")
    # Get features and labels
    train_features, train_labels = preprocessData(trainingData, labelsRow=2)
    # Train the model
    model = train(train_features, train_labels)

    #Predict and evaluate

    #If the test data comes with labels, evaluate
    evaluation = 'aki' in testData.columns

    if evaluation:
        labels_index = testData.columns.get_loc('aki')  # Get the index
        test_features, test_labels = preprocessData(testData, labelsRow=labels_index)
    else:
        test_features, = preprocessData(testData)
    
    # Predict values
    test_predicted = predict(test_features, model)

    #Evaluate
    if evaluation:
        scores = evaluate(test_labels, test_predicted)
        print("f3 score: ", scores)
    
    # Write to files
    w = csv.writer(open(flags.output, "w"))
    w.writerow(("aki",))
    for line in test_predicted:
        w.writerow(line)

if __name__ == "__main__":
    main()