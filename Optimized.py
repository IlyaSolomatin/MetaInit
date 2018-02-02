from config import ALL_FILES
import numpy as np

optimized_qualities = []

FILES = ALL_FILES

#First we should normalize the performance qualities for each dataset
#Pay attention to the directory you read the files from
all_files = []
for file in FILES:
    all_files.append(np.loadtxt('./SVMlinearRBF/'+file[:-5]+'.txt',delimiter=', '))
    #Change the line above to the line below if you work with files which include
    # precomputed time of execution (like XGBonGrid) in the last column
    # all_files.append(np.loadtxt('./XGBonGrid/' + file[:-5] + '.txt', delimiter=', ')[:,:-1])
    if np.max(all_files[-1][:,-1]) != np.min(all_files[-1][:,-1]):
        all_files[-1][:,-1] = (all_files[-1][:,-1] - np.min(all_files[-1][:,-1]))  / (np.max(all_files[-1][:,-1])-np.min(all_files[-1][:,-1]))
    else:
        all_files[-1][:,-1] = np.ones_like(all_files[-1][:,-1])
all_files = np.array(all_files)

for test_file in range(len(FILES)):
    print(test_file)
    best_mean = 0
    for j in range(len(all_files[0])):
        if np.mean(all_files[[i for i in range(len(FILES)) if i != test_file],j,-1]) > best_mean:
            best_mean = np.mean(all_files[[i for i in range(len(FILES)) if i != test_file],j,-1])
            best_config = j
    optimized_qualities.append(1 - all_files[test_file,best_config,-1])

print(np.round(np.mean(optimized_qualities),decimals=3),"±",np.round(np.std(optimized_qualities),decimals=3))