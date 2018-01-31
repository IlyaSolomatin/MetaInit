from config import ALL_FILES
import pickle
import numpy as np

FILES = ALL_FILES

best_closest_qualities = []

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

for test_file in range(len(FILES)):
    closest_file = None
    closest_distance = np.inf
    for file in range(len(FILES)):
        if file != test_file:
            dist = np.linalg.norm(np.array(A[test_file])-np.array(A[file]),ord=2)
            if dist < closest_distance:
                closest_distance = dist
                closest_dataset = file
    best_closest_qualities.append(1-all_files[test_file,np.argmax(all_files[closest_dataset,:,-1]),-1])
print(np.mean(best_closest_qualities))