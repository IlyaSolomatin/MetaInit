import arff
from sklearn.model_selection import cross_val_score
import numpy as np
import scipy
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import timeit
import pickle
from os import listdir
from os.path import isfile, join

#While extracting metafeatures, we follow the paper
#http://aad.informatik.uni-freiburg.de/papers/15-AAAI-MI-SMBO.pdf
#by Feurer et al., 2015
#We extract first 44 of 46 described metafeatures

#Unfortunately, it is impossible to compare the calculated metafeatures with the paper
#In the paper, there were only max/min/average borders for each metafeature on a set of several datasets
#Running this script on the same datasets, we have all the metafeatures falling in +-5% neighborhood of corresponding borders, except:

# Discrepancy of 10.714% in feature class prob mean
# Calculated:  0.036 0.282 0.5
# should be:  0.04 0.28 0.5

# Discrepancy of 6.758% in feature kurtosis max
# Calculated:  -1.3 206.502 4812.487
# should be:  -1.3 193.43 4812.49

# Discrepancy of 190.833% in feature pca 95
# Calculated:  0.026 0.689 2.908
# should be:  0.02 0.52 1.0

# Discrepancy of 361.028% in feature pca skewness first pc
# Calculated:  -5.222 2.633 29.782
# should be:  -27.01 -0.16 6.46

# Discrepancy of 145.288% in feature pca kurtosis first pc
# Calculated:  -1.707 32.82 886.995
# should be:  -2.0 13.38 730.92

def Get_descriptive_vector(file):
    V = []
    times = []

    f = open(file, 'r')
    data = arff.load(f)
    f.close()
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

    start_time = timeit.default_timer()
    #Number of instances
    npatterns = len(data['data'])
    V.append(npatterns)
    #Log number of instances
    V.append(np.log(npatterns))
    times.append(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    #Number of classes
    V.append(len(set(np.asarray(data['data'])[:,-1])))
    times.append(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    #Number of features
    nfeatures = len(data['data'][0])-1
    V.append(nfeatures)
    #Log number of features
    V.append(np.log(nfeatures))
    times.append(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    patterns_with_None = set()
    features_with_None = set()
    missing_values = 0
    for i in range(npatterns):
        for j in range(nfeatures):
            if data['data'][i][j] == None:
                patterns_with_None.add(i)
                features_with_None.add(j)
                missing_values += 1

    #Number of patterns with missing values
    V.append(len(patterns_with_None))
    #Percentage of patterns with missing values
    V.append(len(patterns_with_None)/npatterns)
    #Number of features with missing values
    V.append(len(features_with_None))
    #Percentage of features with missing values
    V.append(len(features_with_None)/nfeatures)
    #Number of missing values
    V.append(missing_values)
    #Percentage of missing values
    V.append(missing_values/(nfeatures*npatterns))
    times.append(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    categorical_cols = 0
    numeric_cols = 0
    categorical_values = []
    kurtosises = []
    skewnesses = []
    list_of_categorical_cols = []
    for i in range(len(data['attributes'])-1):
        if type(data['attributes'][i][1]) != str:
            categorical_cols += 1
            list_of_categorical_cols.append(i)
            distinct_values = set(data['attributes'][i][1]) #set(np.asarray(data['data'])[:,i])
#             try:
#                 distinct_values.remove(None)
#             except KeyError:
#                 pass
            categorical_values.append(len(distinct_values))
        else:
            numeric_cols += 1
            kurtosises.append(scipy.stats.kurtosis(np.asarray(data['data'])[:,i].astype(float),fisher=True,bias=True,nan_policy='omit'))
            skewnesses.append(scipy.stats.skew(np.asarray(data['data'])[:,i].astype(float),bias=False,nan_policy='omit'))

    #Number of numeric features
    V.append(numeric_cols)
    #Number of categorical features
    V.append(categorical_cols)
    #Ratio numerical to categorical
    try:
        V.append(numeric_cols/categorical_cols)
    except ZeroDivisionError:
        V.append(0.)
    #Ratio categorical to numerical
    try:
        V.append(categorical_cols/numeric_cols)
    except ZeroDivisionError:
        V.append(0.)
    #Dataset dimensionality
    V.append(nfeatures/npatterns)
    #Log dataset dimensionality
    V.append(np.log(nfeatures/npatterns))
    #Inverse dataset dimensionality
    V.append(npatterns/nfeatures)
    #Log inverse dataset dimensionality
    V.append(np.log(npatterns/nfeatures))
    times.append(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    class_probs = []
    for c in list(set(np.asarray(data['data'])[:,-1])):
        class_probs.append(list(np.asarray(data['data'])[:,-1]).count(c)/npatterns)

    #Minimal class probability
    V.append(np.min(class_probs))
    #Maximal class probability
    V.append(np.max(class_probs))
    #Average class probability
    V.append(np.mean(class_probs))
    #Std of class probability
    V.append(np.std(class_probs))
    times.append(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    #Class entropy
    V.append(np.sum([-i*np.log2(i) for i in class_probs]))
    times.append(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    if categorical_values != []:
        #Minimal number of categorical values in feature
        V.append(np.min(categorical_values))
        #Maximal number of categorical values in feature
        V.append(np.max(categorical_values))
        #Average number of categorical values in feature
        V.append(np.mean(categorical_values))
        #Std of numbers of categorical values in feature
        V.append(np.std(categorical_values))
        #Total number of categorical values in features
        V.append(sum(categorical_values))
    else:
        V += [0,0,0,0,0]

    if kurtosises != []:
        #Minimal kurtosis of numerical feature
        V.append(np.min(kurtosises))
        # Maximal kurtosis of numerical feature
        V.append(np.max(kurtosises))
        #Average kurtosis of numerical feature
        V.append(np.mean(kurtosises))
        #Std of kurtosises of numerical features
        V.append(np.std(kurtosises))
    else:
        V += [0,0,0,0]

    if skewnesses != []:
        #Minimal skewness of numerical feature
        V.append(np.min(skewnesses))
        #Maximal skewness of numerical feature
        V.append(np.max(skewnesses))
        #Average skewness of numerical feature
        V.append(np.mean(skewnesses))
        #Std of skewnesses of numerical features
        V.append(np.std(skewnesses))
    else:
        V += [0,0,0,0]
    times.append(timeit.default_timer() - start_time + times[-3])

    start_time = timeit.default_timer()
    #Now we have to extract so-called 'landmarking' features
    #But before we should make one-hot encoding of categorical values
    a = pd.DataFrame(data=np.asarray(data['data'])[:,:-1])
    if list_of_categorical_cols != []:
        a = pd.concat([a, pd.get_dummies(a.iloc[:,list_of_categorical_cols])], axis=1, join_axes=[a.index])
        a = a.drop(a.iloc[:,list_of_categorical_cols].head(0).columns, axis=1)
    a = a.replace(to_replace=[None,np.nan,np.inf,'None','none','inf','NaN','nan'],value=0.).apply(pd.to_numeric)

    pca = PCA(n_components=len(list(a)),svd_solver='auto',whiten=False)
#     a = a.replace(to_replace=[None,np.nan,np.inf],value=0.)
    pca.fit(a)

    total_explained_ratio = []
    for i in pca.explained_variance_ratio_:
        total_explained_ratio.append(i)
        if sum(total_explained_ratio) >= 0.95:
            d_prime = len(total_explained_ratio)
            break
    #Ratio of principal components which explain 95% of variance to number of features
    V.append(d_prime/nfeatures)
    #Skewness of the first principal component
    V.append(scipy.stats.skew(pca.components_[0],bias=False))
    #Kurtosis of the first principal component
    V.append(scipy.stats.kurtosis(pca.components_[0],fisher=True,bias=True))
    times.append(timeit.default_timer() - start_time)

    Y = np.asarray(data['data'])[:,-1]

    start_time = timeit.default_timer()
    #From here we append average 10-fold cv accuracy of different simple classifiers
    clf = KNeighborsClassifier(n_neighbors=1)
    V.append(cross_val_score(clf,a,Y,cv=10).mean())
    times.append(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    clf = LDA()
    V.append(cross_val_score(clf,a,Y,cv=10).mean())
    times.append(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    clf = GaussianNB()
    V.append(cross_val_score(clf,a,Y,cv=10).mean())
    times.append(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    clf = DecisionTreeClassifier()
    V.append(cross_val_score(clf,a,Y,cv=10).mean())
    times.append(timeit.default_timer() - start_time)

    return V, times

#Prepare the list of all datasets
files = np.array([f for f in listdir('./Datasets/') if isfile(join('./Datasets/', f)) and f[-5:] == '.arff'])
#A is a list of all metafeature descriptions of our datasets
A = []
for i in range(461):
    #We had some problems with datasets with these indices so we just avoid them
    if i not in [91,117,137,144,328,414,423,129,279,451]:
        a, _ = Get_descriptive_vector('./Datasets/'+files[i])
        print(i)
        A += [a]

# pickle.dump(A, open( "A2.p", "wb" ) )