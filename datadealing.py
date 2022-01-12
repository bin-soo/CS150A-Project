from numpy.core.arrayprint import get_printoptions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import random

train_filepath = 'D:/CS150EVI/CS150A_project/data/train.csv'
traindata = pd.read_table(train_filepath)

def searchkey():
    print(traindata.columns)
    return traindata.columns

def get_first(input_value):
    print(traindata.head(input_value))
    
def get_key_to_value(name):
    print(traindata[name])

def get_unique_number(name):
    num = len(np.unique(traindata[name]))
    print('Number of unique ', name ,' : ', num)
    return num
    
def describle_static_data(name):
    csd = traindata[name]
    #csd.describe()
    print(csd.describe())

def plot_function(name):
    csd = traindata[name]
    hist = plt.hist(np.array(csd.dropna()),bins=100,density=True,log=False,range=(0,100))
    plt.xlabel(name)
    plt.ylabel('Fraction')
    plt.show()
    counts, bins = hist[0], hist[1]
    cdf = np.cumsum(counts)
    plt.plot(bins[1::], cdf)
    plt.xlabel(name)
    plt.ylabel('Cumulative fraction')
    plt.axis((0,100,0,1.0))
    plt.show()

def get_problem_name():
    problems = traindata['Problem Name']
    print(problems)

#plt.subplot2grid((1, 1),(0,0))

#traindata["Anon Student Id"].value_counts().plot(kind = 'bar')
#plt.title("CFA")
#plt.ylabel("number")

names = ["Anon Student Id", "Problem Name", "Step Name", "Correct First Attempt", "Step Duration (sec)"]
corr = traindata[names].corr()
print(corr)