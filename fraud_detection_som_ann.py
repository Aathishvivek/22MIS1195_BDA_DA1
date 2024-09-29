#Making a Hybird Deep Learning Model

#Identify the Frauds with SOM
#Importing libraries

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from pylab import bone, pcolor, colorbar, plot, show
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the training set
dataset = pd.read_csv("Credit_Card_Applications.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
x = sc.fit_transform(x)

#Trainning the SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(x)
som.train_random(data=x, num_iteration=100)

#Visualizing the results
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, j in enumerate(x):
    w = som.winner(j)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

#Finding the Frauds
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(8, 1)], mappings[(6, 8)]), axis=0)
frauds = sc.inverse_transform(frauds)

#Going from unsupervised to supervised

#Creating the matrix of features
customers = dataset.iloc[:, 1:].values

#Creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

#Feature Scaling
sc_x = StandardScaler()
customers = sc_x.fit_transform(customers)

#Making the ANN
#Importing the Keras libraries and packages

#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(units=2, kernel_initializer="uniform",
               activation='relu', input_dim=15))

#Adding the output layer
classifier.add(Dense(units=1, kernel_initializer="uniform", activation='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to the trainning set
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

#Predicitng The Probability of Frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]
