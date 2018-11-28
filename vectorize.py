import csv
import numpy as np
import re
from scipy import sparse
from scipy.stats import uniform
import sys
from sklearn import preprocessing
import pickle

def search_list(my_list, item):
    length = len(my_list)
    for i in range(0,length):
        if(my_list[i] == item):
            return i
    return -1

def get_sum(myList):
    sum = 0
    for i in range(len(myList)):
        sum += myList[i]
    return sum

voc = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','(',')','>','|','&','~']


train_raw = open('project/logical/data/train.txt',"r")

xTrain = []
yTrain = []

xTrainBoW = []
xTrainBoW_concat = []

reader = csv.reader(train_raw)
while True:
    line = next(reader,None)
    if(line is None):
        break
    xTrain.append((line[0],line[1]))
    yTrain.append(line[2])


for i in range(len(xTrain)):
    #temp = ([0] * len(voc),[0] * len(voc))
    temp = np.zeros((2,len(voc)))
    for j in range(2):
        for letter in xTrain[i][j]:
            index = search_list(voc,letter)
            if (index != -1):
                temp[j][index] += 1
    temp[0] /= sum(temp[0])
    temp[1] /= sum(temp[1])
    xTrainBoW.append(temp)
    xTrainBoW_concat.append(np.hstack((temp[0],temp[1])))
   # wr.writerow(temp)
np.save('project/train_BoW_data.npy', xTrainBoW)
np.save('project/train_y_data.npy', yTrain)
np.save('project/train_BoW_concat_data.npy', xTrainBoW_concat)

'''
for i in range(len(imdb_train_data_tuple[0])):
    print("imdb train ", i)
    imdb_train_index_vec.append([])
    for word in imdb_train_data_tuple[0][i].split():
            index = search_list(imdb_vocabulary,word)
            if(( index != -1)):
                imdb_train_index_vec[i].append(index)
'''


