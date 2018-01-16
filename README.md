# MetaInit
This repository is dedicated to the research on initialization of machine learning algorithms with hyperparameters which are learned from metafeatures describing the dataset.

The pipeline of the research is made as follows:

**./Datasets** folder contains 461 datasets which were downloaded from OpenML.org by filtering with conditions:
- <20000 instances
- <500 features
- 2 classes

**Note**: It is important that .arff files downloaded from OpenML.org contain a short description of each feature column. In each file it is said whether a column reflects numerical or categorical feature. This property is used in the code of the project.

Now we have to calculate a metafeature representation of each dataset. This can be made by running **Get_descriptive_vector.py** at each dataset file. After that we can dump the matrix of all metafeature vectors to **A2.p** (this is already done actually).

**Note**: Not all files are succeded in calculating metafeature representation by different reasons. That's why in code we process all files in alphabetical order except for files with indices [91, 117, 137, 144, 328, 414, 423, 279, 129, 451].  

Now we have to run some ML algorithm at all these datasets on some grid of hyperparameters. Accuracies of an ML algorithm are calculated with 10-fold CV and written in corresponding .txt files.

We can make these runs with **SVMlinearRBF.py**, **SVMgammaC.py** or **CatboostOnGrid.py**. You already can see the results of these runs in corresponding folders. The first experiment runs SVM with one hyperparameter of two values: use LinearSVM with default settings *or* SVM with RBF kernel and default settings. The second one runs SVM with RBF kernel with two hyperparameters C and gamma of 399 values overall. The third one adjusts the maximal depth of the tree for Catboost choosing from 11 values.

**Note**: Not all datasets were successfully utilized by these runs by different reasons. That's why after all we will work only with files which: have successfully extracted metafeatures *and* were successfully evaluated by an ML algorith on the whole grid of hyperparameters. Intersection of these two sets gives us 444 datasets. 
