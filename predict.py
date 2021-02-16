import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score,accuracy_score,classification_report

# Splitting data into training/testing
from sklearn.model_selection import train_test_split,GridSearchCV

# Importing libraries for building the neural network
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

X_train=pd.read_csv('xtrain.csv')
X_test=pd.read_csv('xtest.csv')
y_train=pd.read_csv('ytrain.csv')
y_test=pd.read_csv('ytest.csv')
preds=pd.read_csv('preds.csv')


# Evaluate several ml models by training on training set and testing on testing set
def evaluate(X_train, X_test, y_train, y_test):
    # Names of models
    model_name_list = ['Linear Regression',
                      'Random Forest', 'Extra Trees',
                        'Gradient Boosted','KNeighbors']
    # Instantiate the models
    model1 = LinearRegression()
    model3 = RandomForestClassifier(n_estimators=50)
    model4 = ExtraTreesClassifier(n_estimators=50)
    model6 = GradientBoostingClassifier(n_estimators=20)
    model7= KNeighborsClassifier(n_neighbors = 5)
    
    # Dataframe for results
    results = pd.DataFrame(columns=['r2', 'accuracy','squared_error'], index = model_name_list)
    
    # Train and predict with each model
    for i, model in enumerate([model1, model3, model4, model6,model7]):
   
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        # Metrics
        # The coefficient of determination: 1 is perfect prediction
        r2 = r2_score(y_test,predictions)
        preds=np.where(predictions>0.5,1,0)
        df = pd.DataFrame(preds, columns= ['label'])
        df.to_csv(r'preds.csv', index = False)
        # The mean squared error
        squared_error= mean_squared_error(y_test,preds)

        accuracy = accuracy_score(y_test,preds)
        # Insert results into the dataframe
        model_name = model_name_list[i]
        results.loc[model_name, :] = [r2, accuracy,squared_error]
    
    return results
results=evaluate(X_train, X_test, y_train, y_test)
print(results)
####################################Deep learning
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=8))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=10)
eval_model=classifier.evaluate(X_train, y_train)
eval_model
y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)
r2 = r2_score(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print('r2 score is %d ',r2)
print('accuracy score is %d ',accuracy)