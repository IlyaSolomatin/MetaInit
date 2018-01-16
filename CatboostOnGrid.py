import pandas as pd
import arff
import copy
from sklearn.model_selection import cross_val_score
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
depth = range(2,13)
FILES = ['analcatdata_halloffame.arff', 'analcatdata_impeach.arff',
       'analcatdata_japansolvent.arff', 'analcatdata_lawsuit.arff',
       'analcatdata_marketing.arff', 'analcatdata_michiganacc.arff',
       'analcatdata_neavote.arff', 'analcatdata_negotiation.arff',
       'analcatdata_olympic2000.arff', 'analcatdata_reviewer.arff',
       'analcatdata_runshoes.arff', 'analcatdata_seropositive.arff',
       'analcatdata_supreme.arff', 'analcatdata_uktrainacc.arff',
       'analcatdata_vehicle.arff', 'analcatdata_vineyard.arff',
       'analcatdata_whale.arff', 'analcatdata_wildcat.arff', 'anneal.arff',
       'ar1.arff']

le = LabelEncoder()
for file in FILES:
    f = open(file[:-5]+'.txt','w')
    data = arff.load(open('./data/'+file, 'r'))
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
                                  np.array(data['data'])[:, 1].reshape(-1, 1))).tolist()
    if file == 'irish.arff':
        data['attributes'].append(data['attributes'][3])
        del data['attributes'][3]
        data['data'] = np.hstack((np.array(data['data'])[:, list(range(3)) + list(range(4, len(data['data'][0])))],
                                  np.array(data['data'])[:, 1].reshape(-1, 1))).tolist()
    if file == 'wholesale-customers.arff':
        data['attributes'].append(data['attributes'][7])
        del data['attributes'][7]
        data['data'] = np.hstack((np.array(data['data'])[:, list(range(7)) + list(range(8, len(data['data'][0])))],
                                  np.array(data['data'])[:, 1].reshape(-1, 1))).tolist()
    categorical_cols = []
    numeric_cols = []
    for i in range(len(data['attributes'])):
        if type(data['attributes'][i][1]) != str and i != len(data['attributes'])-1: 
            categorical_cols.append(i)
        elif i != len(data['attributes'])-1:
            numeric_cols.append(i)

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
    le.fit(y)
    y = le.transform(y)
    for d in depth:
        clf = CatBoostClassifier(depth=d,logging_level='Silent')
        f.write(str(d)+', '+str(cross_val_score(clf,X,y,cv=10,scoring='accuracy',n_jobs=10).mean())+'\n')
    f.close()
