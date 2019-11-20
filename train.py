import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor


def compute_loss(y, Y_pred):
    return ((y - Y_pred) ** 2) / 2


def loss_gradient(y, Y_pred):
    return -(y-Y_pred)


def gradient_boost_mse(X, y, M, learning_rate):
    regressors = []
    Y_pred = np.array([y.mean()]*len(y))
    f0 = Y_pred
    print(compute_loss(y, Y_pred).mean())
    for i in range(M):
        residuals = -loss_gradient(y, Y_pred)
        regressor = DecisionTreeRegressor(max_depth=1)
        regressor.fit(X, residuals)
        regressors.append(regressor)
        predictions = regressor.predict(X)
        Y_pred = Y_pred + learning_rate * predictions
        if i % 20 == 0:
            print("Loss after iteration " + str(i) + ": " + str(compute_loss(y, Y_pred).mean()))
    return regressors, f0


def gradient_boost_mse_predict(regressors, f0, X, learning_rate):
    Y_pred = np.array([f0[0]]*len(X))
    for regressor in regressors:
        Y_pred = Y_pred + learning_rate * regressor.predict(X)
    return Y_pred




dataset = pd.read_csv('mushrooms.csv')

print(dataset.head())

#Define the column names
dataset.columns = ['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing',
'gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring',
'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population',
'habitat']

for label in dataset.columns:
    dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label])

le = LabelEncoder()
dataset = dataset.apply(le.fit_transform)

X = dataset.drop(['class'], axis=1)
Y = dataset['class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()
X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)



Y_pred = np.array([Y.mean()] * len(Y))
print(compute_loss(Y, Y_pred).mean())


losses = []
for i in range(-200, 200):
    losses.append(compute_loss(Y[0], Y_pred[0] + i / 100))

print((compute_loss(Y[0], Y_pred[0] + 1) - compute_loss(Y[0], Y_pred[0])) / 1)              #-0.017971442639
print((compute_loss(Y[0], Y_pred[0] + .5) - compute_loss(Y[0], Y_pred[0])) / .5)            #-0.267971442639
print((compute_loss(Y[0], Y_pred[0] + .1) - compute_loss(Y[0], Y_pred[0])) / .1)            #-0.467971442639
print((compute_loss(Y[0], Y_pred[0] + .01) - compute_loss(Y[0], Y_pred[0])) / .01)          #-0.512971442639
print((compute_loss(Y[0], Y_pred[0] + .001) - compute_loss(Y[0], Y_pred[0])) / .001)        #-0.517471442639
print((compute_loss(Y[0], Y_pred[0] + .0001) - compute_loss(Y[0], Y_pred[0])) / .0001)      #-0.517921442639
print((compute_loss(Y[0], Y_pred[0] + .00001) - compute_loss(Y[0], Y_pred[0])) / .00001)    #-0.517966442636
print((compute_loss(Y[0], Y_pred[0] + .000001) - compute_loss(Y[0], Y_pred[0])) / .000001)  #-0.517970942581

residuals = -loss_gradient(Y, Y_pred)


regressor = DecisionTreeRegressor(max_depth=1)
regressor.fit(X, residuals)



print(regressor.tree_.feature)          #[ 8 -2 -2]
print(regressor.tree_.threshold)        #[ 3.5 -2.  -2. ]


leaf1_index = np.where(X.iloc[:, 8] >= 3.5)
leaf2_index = np.where(X.iloc[:, 8] < 3.5)


print(np.unique(regressor.predict(X)))      #[-0.24199533  0.35231243]



regressors, f0 = gradient_boost_mse(X_train, Y_train, 5000, 0.1)
print("Loss: " + str(compute_loss(Y_test, gradient_boost_mse_predict(regressors, f0, X_test, 0.1)).mean()))
results = gradient_boost_mse_predict(regressors, f0, X_test, 0.1)

correct = 0
total = 0
for index, val in enumerate(results):
    if round(val) == Y_test[index]:
        correct = correct + 1
    total = total + 1

print("Accuracy: " + str((correct/total)*100))


#ref: https://datatuts.com/gradient-boosting-in-python-from-scratch/