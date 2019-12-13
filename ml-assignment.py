import csv
import sys
import statistics
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import sklearn as sk
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

featurenames = ["Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3 ","Power_range_sensor_4","Pressure _sensor_1", "Pressure _sensor_2", "Pressure _sensor_3", "Pressure _sensor_4", "Vibration_sensor_1","Vibration_sensor_2", "Vibration_sensor_3", "Vibration_sensor_4"]

data = 'nucleardataset.csv'
data = pd.read_csv(data)
NormalData = data['Status']=="Normal"
NormalData = data[NormalData]
AbnormalData = data['Status']=="Abnormal"
AbnormalData = data[AbnormalData]

def preProcessing(data):
    #Check if the data frame contains missing values.
    nullVal = data.isnull().values.any()
    #If Missing values are true, return a list of columns which contain
    #missing values, as well as print a total of missing values.
    if nullVal == True:
        print("Data contains missing values\n" + str(data.isnull().sum()) + "\nTotal number of missing values : " + str(data.isnull().sum().sum()))
        #If values are missing, replace with the median of the data. This could be 
        #more specific if need be.
        median = data.median()
        data.fillna(median, inplace=True)

    #data = preprocessing.normalize(data)
    data = preprocessing.scale(data)
    return data

def defineVariables(name):
    X = pd.read_csv(name, header=None)
    X = X.drop(X.columns[0], axis=1)
    X = X.drop(X.index[0])
    y = data[data.columns[0]]
    return X, y

def plotData(x, y, cv, title, tag):
    xstr = []
    for i in x:
        xstr.append(str(i) + " " + tag)
    plt.title(title)
    plt.plot(xstr, y ,label='Testing Accuracy')
    plt.plot(xstr, cv, label='10-fold CV Accuracy')
    plt.legend()
    plt.show()

#Prints a density Plot using seaborn package. Using spilt values previously, the normal and abnormal data are
#shown using kdeplot.
def densityPlot(NormData, AbnoData):
    sns.kdeplot(NormData, shade=True, color="r", label="Normal")
    sns.kdeplot(AbnoData, shade=True, color="b", label="Abnormal")
    plt.title("Vibration_sensor_2”")
    plt.show()

#Prints a box plot using matplot package
def boxPlot(NormData, AbnoData):
    box_plot_data=[NormData, AbnoData]
    plt.title("Vibration_sensor_1")
    plt.boxplot(box_plot_data, labels=['Normal', 'Abnormal'])
    plt.show()

#Prints a summary of data using the descirbe function. These include count, mean, std, min, 25%, 50%, 75%, max.
def summaryOfDataSet():
    df_marks = data.describe()
    new_row = []
    #Obtains the size of data in each row using sys.getsizeof function
    for f in featurenames:
        sizeOfData = sys.getsizeof(data[f])
        new_row.append(sizeOfData)
    df_marks.loc['Size of Data'] = new_row
    #Writes data within an excel file
    writer = pd.ExcelWriter('output.xlsx')
    df_marks.to_excel(writer)
    writer.save()

def accuracy(confusionMatrix):
    return (confusionMatrix[0][0] + confusionMatrix[1][1])/(confusionMatrix[0][0] + confusionMatrix[0][1] + confusionMatrix[1][0] + confusionMatrix[1][1])

def randomForest(n_trees, cv, min_leaf, X, y):
    xtrain, xtest, ytrain, ytest = splitData(X, y)
    print("Fitting Model(s) : Random Forest")
    for leaf in min_leaf:
        CVaccuracyscore = []
        predictions = []
        accuracyscore = []
        cross_score = []
        index = 0 
        for i in n_trees:
            RF = RandomForestClassifier(n_estimators=i, random_state=0, min_samples_leaf=leaf)
            RF.fit(xtrain, ytrain)
            predictions.append(RF.predict(xtest))
            k_fold_validation = model_selection.KFold(n_splits=cv, random_state=12)
            cross_score.append(cross_val_score(RF, xtrain, ytrain, cv=k_fold_validation, scoring='accuracy'))
        
        for i in predictions:
            print("\nParameters:\nNumber of Trees: " + str(n_trees[index]) + "\nNumber of samples per leaf: " + str(leaf))
            print("Accuracy: ", str(metrics.accuracy_score(ytest, i)))
            print("CV Mean accuracy: + " + str(cross_score[index].mean()) +" (+/- " + str(cross_score[index].std() * 2) + ")\n")
            accuracyscore.append(metrics.accuracy_score(ytest, i))
            CVaccuracyscore.append(cross_score[index].mean())
            index = index+1
        
        plotData(n_trees, accuracyscore, CVaccuracyscore, "Random Forest with a minimum of " + str(leaf) + " samples per leaf node", "trees")

# some parameter combinations will not converge, so this will ignore them
@ignore_warnings(category=ConvergenceWarning)

def nerualNetwork(hiddenLayers, cv, maxIter, X, y):
    xtrain, xtest, ytrain, ytest = splitData(X, y)
    CVaccuracyscore = []
    accuracyscore = []
    for it in maxIter:
        for i in hiddenLayers:
            mlp = MLPClassifier(hidden_layer_sizes=i,activation="logistic", max_iter=it, solver='adam')
            print("\nFitting Model : Artifcal Neural Network : " + str(i) + "\n")
            mlp.fit(xtrain,ytrain) ; predictions = mlp.predict(xtest)
            print(classification_report(ytest,predictions) +"\nOverall accuracy of model (True Positive + True Negative / Total) : " + str(accuracy(confusion_matrix(ytest, predictions))) + "\n\nCalculating Cross Validation Scores")
            k_fold_validation = model_selection.KFold(n_splits=cv, random_state=12)
            cross_score = cross_val_score(mlp, xtrain, ytrain, cv=k_fold_validation, scoring='accuracy')
            print("\nParamaters\nNumber of hidden layers: " + str(len(hiddenLayers)) + "\nNumber of hidden neurons: " + str(i) + "\nEpochs: " + str(it))
            print("\n" + str(len(cross_score)) + "-fold CV")
            print("Mean accuracy: %0.2f (+/- %0.2f)" % (cross_score.mean(), cross_score.std() * 2))
            accuracyscore.append(accuracy(confusion_matrix(ytest, predictions)))
            CVaccuracyscore.append(cross_score.mean())
        #Plot data for several layers
        #plotData(hiddenLayers, accuracyscore, CVaccuracyscore, "Neural Network with " + str(it) + "epochs ", "neruons")
    #Plot Data for one hidden layer and multiple itterations
    plotData(maxIter, accuracyscore, CVaccuracyscore, "Neural Network with " + str(hiddenLayers) + "neurons ", "epoch")

    
def splitData(X, y):
    xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size=0.1, shuffle=True)
    return xtrain, xtest, ytrain, ytest

def main():

    #General Variables
        #Cross Validation: The number of folds to pass to the random forest or Neural Network
    fileName = 'nucleardataset.csv'
    crossVal = 10

    #Neural Network variables:
        #Hidden Layer : Can contain a list or indvidual variables. (X, X) is two hidden layers.
        #Max Interations : The number of times the neural network will train
    hiddenLayer = [(500,500)]
    maxIter = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    #RandomForest variables
        #The number of trees
        #The number of leafs
    trees = [10, 30, 50, 100, 200, 500, 1000]
    minLeaf = [10, 20, 30, 40]

    #Summary Stage
    summaryOfDataSet()

    #Visualisation
    #boxPlot(NormalData[featurenames[8]], AbnormalData[featurenames[8]])
    #densityPlot(NormalData[featurenames[9]], AbnormalData[featurenames[9]])

    #Preprocessing Stage
    X, y = defineVariables(fileName)
    X = preProcessing(X)

    nerualNetwork(hiddenLayer, crossVal, maxIter, X, y)
    #randomForest(trees, crossVal, minLeaf, X, y)
    
main()
