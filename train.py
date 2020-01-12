# ref: https://www.kaggle.com/ankitkuls/xgboost-with-one-hot-encoding
# ref: https://datatuts.com/gradient-boosting-in-python-from-scratch/
# ref: https://machinelearningmastery.com/evaluate-gradient-boosting-models-xgboost-python/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('mushrooms.csv')

print(dataset.head())

#Define the column names
dataset_columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                   'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                   'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                   'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

data_columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                   'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                   'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                   'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
Y = dataset['class']
X = dataset.drop(['class'], axis=1)

# encode string values as integers for data then use one-hot encoding

encoded_X = None
for i in range(0, X.shape[1]):
    label_encoder = LabelEncoder()
    feature = label_encoder.fit_transform(X.iloc[:,i])
    feature = feature.reshape(X.shape[0], 1)
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    feature = onehot_encoder.fit_transform(feature)
    if encoded_X is None:
        encoded_X = feature
    else:
        encoded_X = np.concatenate((encoded_X, feature), axis=1)

# encode string values as integers for labels
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
encoded_Y = label_encoder.transform(Y)


# split train, validation and test datasets
X_train, X_testval, Y_train, Y_testval = train_test_split(encoded_X, encoded_Y, test_size=0.2)
X_test, X_val, Y_test, Y_val = train_test_split(X_testval, Y_testval, test_size=0.5)


# shapes of data
print("X_train.shape: " + str(X_train.shape))
print("Y_train.shape: " + str(Y_train.shape))
print("X_val.shape: " + str(X_val.shape))
print("Y_val.shape: " + str(Y_val.shape))
print("X_test.shape: " + str(X_test.shape))
print("Y_test.shape: " + str(Y_test.shape))

# Implementation of the scikit-learn API for XGBoost classification.
xg_model = XGBClassifier(n_estimators=1000, learning_rate=0.05)

# Fit gradient boosting model
xg_model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=False)

# Predict with data
test_preds = xg_model.predict(X_test)

#round predictions
predictions = [round(value) for value in test_preds]

# evaluate predictions
accuracy = accuracy_score(Y_test, predictions)

#print accuracy
print("Accuracy: %.2f%%" % (accuracy * 100.0))
