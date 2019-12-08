import csv
import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import sys
import numpy as np
import keras 
import statistics
import matplotlib.pyplot as plt
from sklearn import preprocessing

featurenames = ["Power_range_sensor_1", "Power_range_sensor_2", "Power_range_sensor_3 ","Power_range_sensor_4","Pressure _sensor_1", "Pressure _sensor_2", "Pressure _sensor_3", "Pressure _sensor_4", "Vibration_sensor_1","Vibration_sensor_2", "Vibration_sensor_3", "Vibration_sensor_4"]

data = 'nucleardataset.csv'
data = pd.read_csv(data)

NormalData = data['Status']=="Normal"
NormalData = data[NormalData]
AbnormalData = data['Status']=="Abnormal"
AbnormalData = data[AbnormalData]

#Prints a density Plot using seaborn package
def densityPlot(NormData, AbnoData):
    sns.kdeplot(NormData, shade=True, color="r", label="Normal")
    sns.kdeplot(AbnoData, shade=True, color="b", label="Abnormal")
    plt.show()

#Prints a box plot using matplot package
def boxPlot(NormData, AbnoData):
    box_plot_data=[NormData, AbnoData]
    plt.boxplot(box_plot_data)
    plt.show()

#Prints a summary of data using the descirbe function. These include count, mean, std, min, 25%, 50%, 75%, max.
def summaryOfDataSet():
    desc = data.describe()
    for item in featurenames:
        print("Summary of : " + str(item))
        print("size     " + str(sys.getsizeof(data[item])))
        print(desc[item]); print("\n")

def normalization(data):
    data = preprocessing.normalize(data)
    print(data)

def randomForest(data):
    y = data.iloc[:,12]
    X = data.iloc[:,:12]
    RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    RF.fit(X, y)
    RF.predict(X.iloc[900:,:])
    round(RF.score(X,y), 4)




def main():
    summaryOfDataSet()
    print("size of dataset  " + str(sys.getsizeof(data)))
    print("Number of features   " + str(len(data)))
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(['Normal', 'Abnormal'])
    

    #boxPlot(NormalData[featurenames[8]], AbnormalData[featurenames[8]])
    #densityPlot(NormalData[featurenames[9]], AbnormalData[featurenames[9]])
    #normalization(data[1:])
    #randomForest(data)

main()
