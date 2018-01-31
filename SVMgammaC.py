import pandas as pd
import arff
import copy
from config import ALL_FILES
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

np.random.seed(42)
C_range = list(range(-5,16))
gamma_range = list(range(-15,4))
#It is highly recommended to parallel this script by running it on different files at the same time
FILES = ALL_FILES

for file in FILES:
    f = open(file[:-5]+'.txt','w')
    data = arff.load(open('./data/'+file, 'r'))
       
    #This ugly block is here because in some datasets downloaded from OpenML the target column is not the last one.
    #It forces to write a lot of exceptions like these to move the target column to the last place in order to
    #process all the files in the same way with target column in behind.
    if file in ['prnn_crabs.arff','profb.arff','sleuth_ex2015.arff','sleuth_ex2016.arff','analcatdata_asbestos.arff',
                'Australian.arff','dataset_106_molecular-biology_promoters.arff','dataset_114_shuttle-landing-control.arff',
                'kdd_internet_usage.arff','molecular-biology_promoters.arff','monks-problems-1.arff','monks-problems-2.arff',
                'monks-problems-3.arff','oil_spill.arff','SPECT.arff']:
        data['attributes'].append(data['attributes'][0])
        del data['attributes'][0]
        data['data'] = np.hstack((np.array(data['data'])[:, 1:], np.array(data['data'])[:, 0].reshape(-1, 1))).tolist()
    if file == 'analcatdata_whale.arff':
        data['data'] = data['data'][:-5]
    if file in ['analcatdata_japansolvent.arff','lungcancer_GSE31210.arff','lupus.arff',]:
        data['attributes'].append(data['attributes'][1])
        del data['attributes'][1]
        data['data'] = np.hstack((np.array(data['data'])[:, [0] + list(range(2, len(data['data'][0])))],
                                  np.array(data['data'])[:, 1].reshape(-1, 1))).tolist()
    if file == 'dataset_25_colic.ORIG.arff':
        data['attributes'].append(data['attributes'][23])
        del data['attributes'][23]
        data['data'] = np.hstack((np.array(data['data'])[:, list(range(23)) + list(range(24, len(data['data'][0])))],
                                  np.array(data['data'])[:, 23].reshape(-1, 1))).tolist()
    if file == 'irish.arff':
        data['attributes'].append(data['attributes'][3])
        del data['attributes'][3]
        data['data'] = np.hstack((np.array(data['data'])[:, list(range(3)) + list(range(4, len(data['data'][0])))],
                                  np.array(data['data'])[:, 3].reshape(-1, 1))).tolist()
    if file == 'wholesale-customers.arff':
        data['attributes'].append(data['attributes'][7])
        del data['attributes'][7]
        data['data'] = np.hstack((np.array(data['data'])[:, list(range(7)) + list(range(8, len(data['data'][0])))],
                                  np.array(data['data'])[:, 7].reshape(-1, 1))).tolist()
    #Here we understand which features are categorical and which are numerical
    categorical_cols = []
    numeric_cols = []
    for i in range(len(data['attributes'])):
        if type(data['attributes'][i][1]) != str and i != len(data['attributes'])-1: 
            categorical_cols.append(i)
        elif i != len(data['attributes'])-1:
            numeric_cols.append(i)
       
    #Here we make one hot encoding of categorical features and normalize numerical features
    data = pd.DataFrame(data=data['data'],index=None)
    for categorical_col in categorical_cols:
        col = copy.deepcopy(data[categorical_col])
        del data[categorical_col]
        data = pd.concat([pd.get_dummies(col),data],axis=1)
    for numeric_col in numeric_cols:
        data[numeric_col] = pd.to_numeric(data[numeric_col])
        if data[numeric_col].max() - data[numeric_col].min() != 0:
            data[numeric_col] = (data[numeric_col] - data[numeric_col].min()) / (data[numeric_col].max() - data[numeric_col].min())
        else:
            data[numeric_col] = 0.
    data = data.sample(frac=1) #shuffle rows
    data = data.reset_index(drop=True)
    X = data.iloc[:, :-1].fillna(0)
    y = data.iloc[:,-1]
    for C in C_range:
        for gamma in gamma_range:
            clf = SVC(C=2**C,gamma=2**gamma)
            f.write(str(C)+', '+str(gamma)+', '+str(cross_val_score(clf,X,y,cv=10,scoring='accuracy',n_jobs=10).mean())+'\n')
    f.close()
