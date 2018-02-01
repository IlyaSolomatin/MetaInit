from config import ALL_FILES
import numpy as np
import pickle
import xgboost as xgb

FILES = ALL_FILES

#That is the size of the grid of hyperparameters (number of all evaluated combinations of hyperparameters)
#399 for SVMgammaC, 2 for SVMlinearRBF, 11 for Catboost
gridsize = 399

#First we should normalize the performance qualities for each dataset
#Pay attention to the directory you read the files from
all_files = []
for file in FILES:
    all_files.append(np.loadtxt('./SVMgammaC/'+file[:-5]+'.txt',delimiter=', '))
    #Change the line above to the line below if you work with files which include
    #precomputed time of execution (like XGBonGrid) in the last column
    #all_files.append(np.loadtxt('./XGBonGrid/' + file[:-5] + '.txt', delimiter=', ')[:,:-1])
    if np.max(all_files[-1][:,-1]) != np.min(all_files[-1][:,-1]):
        all_files[-1][:,-1] = (all_files[-1][:,-1] - np.min(all_files[-1][:,-1]))  / (np.max(all_files[-1][:,-1])-np.min(all_files[-1][:,-1]))
    else:
        all_files[-1][:,-1] = np.ones_like(all_files[-1][:,-1])
all_files = np.array(all_files)

#Now we load the matrix with metafeature vectors for all files
A = np.array(pickle.load(open("descriptive_vectors.p","rb")))
first = True
#Then we construct (X,Y) which consist of metafeatures of files, evaluated hyperparameters and measured qualities
for file in range(len(FILES)):
    if first:
        a = np.hstack((np.array([A[file] for i in range(gridsize)]), all_files[file]))
        first = False
    else:
        buf = np.hstack((np.array([A[file] for i in range(gridsize)]), all_files[file]))
        a = np.vstack((a, buf))

xgb_qualities = []
print(len(FILES))
for test_file in range(len(FILES)):
    print(test_file,np.mean(xgb_qualities))
    #Here we throw out a part of (X,Y) which is related to test file since we should not train on it
    b = np.vstack((a[:len(all_files[0])*test_file], a[len(all_files[0])*(test_file+1):]))
    clf = xgb.XGBRegressor(nthread=2)
    clf.fit(b[:,:-1],b[:,-1])
    Grid = all_files[0, :, :-1]
    #Here we construct a grid for which we will predict the performance. It consists of metafeatures of test file and grid of hyperparameters
    Grid = np.hstack((np.array([A[test_file] for _ in range(gridsize)]),Grid))
    prediction = clf.predict(Grid)
    best_config = np.argmax(prediction)
    xgb_qualities.append(1-all_files[test_file,best_config,-1])

print(np.mean(xgb_qualities))