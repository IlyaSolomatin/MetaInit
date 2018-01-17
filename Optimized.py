from config import Get_FILES#,Get_FILES_Cat Get_FILES_Cat
import numpy as np
import pickle

optimized_qualities = []
optimized_raw_qualities = []

#Uncomment the line with Get_FILES_Cat() and comment the line with Get_FILES() in order to work with files which are
#computed with CatBoost, not SVM
#FILES = Get_FILES_Cat()
FILES = Get_FILES()
# print(len(FILES))

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
# print(len(all_files[0]))
for test_file in range(len(FILES)):
    print(test_file)
    best_mean = 0
    for j in range(len(all_files[0])):
        if np.mean(all_files[[i for i in range(len(FILES)) if i != test_file],j,-1]) > best_mean:
            best_mean = np.mean(all_files[[i for i in range(len(FILES)) if i != test_file],j,-1])
            best_config = j
    optimized_qualities.append(1 - all_files[test_file,best_config,-1])
#     optimized_raw_qualities.append(np.loadtxt('./data3/'+FILES[test_file]+'.txt',delimiter=', ')[best_config,-1])
# pickle.dump(optimized_raw_qualities, open( "optimized_raw_qualities.p", "wb" ) )
# pickle.dump(optimized_qualities, open( "optimized_qualities.p", "wb" ) )
print(np.mean(optimized_qualities))