import pandas as pd
df = pd.read_csv('diabetes.csv')

df.hist()

from matplotlib import pyplot as plt
plt.show()

import seaborn as sns

#Create a subplot of 3 x 3 
figure, axes = plt.subplots(3,3,figsize=(15,15))

#Make sure there is enough padding to allow titles to be seen
figure.tight_layout(pad=5.0)

#Plot a density plot for each variable
for idx, col in enumerate(df.columns):
    ax = plt.subplot(3, 3, idx + 1)
    ax.yaxis.set_ticklabels([])
    sns.distplot(df.loc[df.Outcome == 0][col], hist=False, axlabel=False, kde_kws={'linestyle':'-', 'color':'black', 'label':"No Diabetes"})
    sns.distplot(df.loc[df.Outcome == 1][col], hist=False,
    axlabel= False, kde_kws={'linestyle':'--',
    'color':'black', 'label':"Diabetes"})
    ax.set_title(col)

# Hide the 9th subplot (bottom right) since the relationship between the
# two outcomes themselves is meaningless
plt.subplot(3,3,9).set_visible(False)
# Show the plot
plt.show()

print(df.isnull().any())

pd.set_option('max_columns', None)
print(df.describe(include='all'))

print("Number of rows with 0 values for each variable:")
for col in df.columns:
    missing_rows = df.loc[df[col] == 0].shape[0]
    print(col + ": " + str(missing_rows))

import numpy as np

df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)

df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

from sklearn import preprocessing

# Normalize the data via centering
# Use the scale() function from scikit-learn
print("Centering the data...")
df_scaled = preprocessing.scale(df)

# Result must be converted back to a pandas DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Do not want the Outcome column to be scaled, so keep the original
df_scaled['Outcome'] = df['Outcome']
df = df_scaled

print(df.describe().loc[['mean', 'std','max'],].round(2).abs())

from sklearn.model_selection import train_test_split

# Split dataset into an input matrix (all columns but Outcome) and Outcome vector
X = df.loc[:, df.columns != 'Outcome']
y = df.loc[:, 'Outcome']

# Split input matrix to create the training set (80%) and testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Second split on training set to create the validation set (20% of training set)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

from tensorflow import keras
from tensorflow.keras.models import Sequential
#from keras import Sequential
model = Sequential()

#add the first hidden layer
from tensorflow.keras.layers import Dense
#number of neurons, activation function, number of inputs each neuron receives
#model.add(Dense(32, activation='relu', input_dim=8))
model.add(Dense(2, activation='relu', input_dim=8))

#add the second hidden layer
#model.add(Dense(16, activation='relu'))
#model.add(Dense(4, activation='relu'))

#add the third hidden layer
#model.add(Dense(8, activation='relu'))

#add the fourth hidden layer
#model.add(Dense(2, activation='relu'))

#add the fifth hidden layer
#model.add(Dense(2, activation='relu'))

#add the output layer
model.add(Dense(1, activation='sigmoid'))

#compile the network
#Optimizer, loss function, metrics to evaluate the network
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train the network
#model.fit(X_train, y_train, batch_size=1, epochs=200)
model.fit(X_train, y_train, epochs=200)

#Evaluate the accuracy with respect to the training set
scores = model.evaluate(X_train, y_train)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))

#Evaluate the accuracy with respect to the testing set
scores = model.evaluate(X_test, y_test)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

from sklearn.metrics import confusion_matrix

#Construct a confusion matrix to evaluate the performance of the model
y_test_pred = (model.predict(X_test) > 0.5).astype("int32")
c_matrix = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(c_matrix, annot=True, xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'], cbar=False, cmap="Blues")  # annot=True to annotate cells
ax.set_xlabel('Prediction')
ax.set_ylabel('Actual')

plt.show()

from sklearn.metrics import roc_curve

y_test_pred_probs = model.predict(X_test)

FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)

plt.plot(FPR, TPR)
plt.plot([0, 1], [0, 1], '--', color='black') #diagonal line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()