import pandas as pd
import arff
import copy
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK
import timeit
from config import ALL_FILES
import pickle

REPEATS = 10
MAX_EVALS = 20

space = {
    'C': hp.choice('C', range(-5,16)),
    'gamma': hp.choice('gamma', range(-15,4))}

# np.random.seed(42)
#It is highly recommended to parallel this script by running it on different files at the same time
FILES = ALL_FILES

def objective(params):
    global file, times, qualities
    data = arff.load(open('./Datasets/'+file, 'r'))
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
    start = timeit.default_timer()
    clf = SVC(C=2**params['C'],gamma=2**params['gamma'])
    score = cross_val_score(clf,X,y,cv=10,scoring='accuracy',n_jobs=10).mean()
    t = timeit.default_timer()-start
    if times != []:
        times.append(times[-1]+t)
    else:
        times.append(t)
    if qualities != []:
        if score > qualities[-1]:
            qualities.append(score)
        else:
            qualities.append(qualities[-1])
    else:
        qualities.append(score)
    return {'loss': -score, 'status': STATUS_OK }

Q = []
T = []
for _ in range(REPEATS):
    all_qualities = []
    all_times = []
    for file in FILES:
        times = []
        qualities = []
        best = fmin(objective,space=space,algo=tpe.suggest,max_evals=MAX_EVALS)
        all_qualities.append(qualities)
        all_times.append(times)
    Q.append(all_qualities)
    T.append(all_times)
    
# pickle.dump(T,open("T.p","wb"))
# pickle.dump(Q,open("Q.p","wb"))