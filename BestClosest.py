from config import ALL_FILES, Get_FILES#,Get_FILES_Cat
import pickle
import numpy as np

META_FILES = [ALL_FILES[i][:-5] for i in range(len(ALL_FILES)) if
              i not in [91, 117, 137, 144, 328, 414, 423, 279, 129, 451]]

#Uncomment the line with Get_FILES_Cat() and comment the line with Get_FILES() in order to work with files which are
#computed with CatBoost, not SVM
# FILES = Get_FILES_Cat()
FILES = Get_FILES()

best_closest_qualities = []

#First we should normalize the performance qualities for each dataset
#Pay attention to the directory you read the files from
all_files = []
for file in FILES:
    all_files.append(np.loadtxt('./SVMgammaC/'+file+'.txt',delimiter=', '))
    if np.max(all_files[-1][:,-1]) != np.min(all_files[-1][:,-1]):
        all_files[-1][:,-1] = (all_files[-1][:,-1] - np.min(all_files[-1][:,-1]))  / (np.max(all_files[-1][:,-1])-np.min(all_files[-1][:,-1]))
    else:
        all_files[-1][:,-1] = np.zeros_like(all_files[-1][:,-1])
all_files = np.array(all_files)

#Now we load the matrix with metafeature vectors for all files
A = np.array(pickle.load(open("A2.p","rb")))

for test_file in range(len(FILES)):
    closest_file = None
    closest_distance = np.inf
    for file in range(len(FILES)):
        if file != test_file:
            dist = np.linalg.norm(np.array(A[META_FILES.index(FILES[test_file])])-np.array(A[META_FILES.index(FILES[file])]),ord=2)
            if dist < closest_distance:
                closest_distance = dist
                closest_dataset = file
    best_closest_qualities.append(1-all_files[test_file,np.argmax(all_files[closest_dataset,:,-1]),-1])
print(np.mean(best_closest_qualities))