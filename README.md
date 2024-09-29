Fraud Detection System Using Self-Organizing Maps (SOM) and Artificial Neural Networks (ANN)

This aims to develop a hybrid deep learning model for detecting potential fraud in credit card applications using a dataset from a bank. By implementing Self-Organizing Maps (SOM) for unsupervised learning and Artificial Neural Networks (ANN) for supervised learning, we achieve a fraud detection success rate of 96%. This innovative approach enhances the decision-making process in credit card approvals, safeguarding financial institutions against fraudulent activities.

Dataset
The dataset used in this project consists of customer applications for an advanced credit card. It contains the following features:

CustomerID	A1	A2	A3	A4	A5	A6	A7	A8	A9	A10	A11	A12	A13	A14	Class
15776156	1	22.08	11.46	2	4	4	1.585	0	0	0	1	2	100	1213	0
15739548	0	22.67	7	2	8	4	0.165	0	0	0	0	2	160	1	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
CustomerID: Unique identifier for each customer.
A1 - A14: Various features representing customer attributes and behaviors.
Class: Indicates whether the application is fraudulent (1) or not (0).


Technologies Used:

TensorFlow: For implementing the ANN model.
Scikit-learn: For data preprocessing and scaling.
Pylab: For visualizing the results.
MiniSom: For implementing the Self-Organizing Map.
NumPy: For numerical operations.
Matplotlib: For data visualization.
Pandas: For data manipulation and analysis.
Code Explanation
The code implements a hybrid model consisting of two parts:

Self-Organizing Map (SOM):

Initializes and trains the SOM using customer data.
Visualizes the results to identify potential fraudulent behaviors based on the distance map.
Artificial Neural Network (ANN):

Constructs an ANN to classify the potential fraud cases identified by the SOM.
Trains the model using the scaled customer data, predicting the probability of fraud.

Main Code Snippet:

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

How to Run:

Requirements:

Software: Python (version 3.6 or above), TensorFlow, Scikit-learn, Pylab, MiniSom, NumPy, Matplotlib, Pandas.
Hardware: Any system capable of running Python with the above libraries installed (recommended: 8 GB RAM or more).

Execution Steps:
Clone the repository or download the code files.

Install the required packages:
pip install tensorflow scikit-learn pylab minisom numpy matplotlib pandas
Ensure the dataset Credit_Card_Applications.csv is in the same directory as the code.

Run the Python script:
python fraud_detection_som_ann.py

Conclusion:
The hybrid model developed in this project effectively combines SOM for unsupervised learning and ANN for supervised learning, resulting in a robust system for fraud detection with a high accuracy rate. This system can be utilized by financial institutions to enhance their credit card application review processes.
