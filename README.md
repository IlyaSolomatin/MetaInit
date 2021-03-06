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

We can make these runs with **SVMlinearRBF.py**, **SVMgammaC.py** or **CatboostOnGrid.py**. You already can see the results of these runs in corresponding folders. 
- The first experiment runs SVM with one hyperparameter of two values: use LinearSVM with default settings *or* SVM with RBF kernel and default settings. 
- The second one runs SVM with RBF kernel with two hyperparameters C and gamma of 399 values overall. 
- The third one adjusts the maximal depth of the tree for Catboost choosing from 11 values.

**Note**: Not all datasets were successfully utilized by these runs by different reasons. That's why after all we will work only with files which: have successfully extracted metafeatures *and* were successfully evaluated by an ML algorith on the whole grid of hyperparameters. Intersection of these two sets gives us 444 datasets. 

After we get the results in format "Grid of hyperparameters: ML algorithm performance" for each dataset, we can run a strategy of hyperparameter selection on this. We can make this experiments with **BestClosest.py**, **Optimized.py** and **XGBdefault.py**.
- The first strategy suggests the hyperparameters configuration which was the best on the dataset, which is the closest to the given one in metafeature space by Euclidean distance.
- The second strategy suggests hyperparameters configuration which maximizes the average performance of an ML algorithm on all the datasets that we have except for the given one.
- The third one fits a surrogate such as XGBoost with default settings on (X,Y) where X is (Metafeatures of the dataset,Hyperparameter configuration) and Y is the quality of an ML algorithm on a dataset with such metafeatures and hyperparameters. We fit it on all the datasets and configurations except for the given dataset. After that we are able to predict the performance of an ML algorithm on a given dataset for the whole grid of hyperparameters using its metafeature description. Taking the maximum of predicted performance we get the best recommended configuration.

After the experiment, strategy scripts report their quality by metric Average Distance to The Maximum (ADTM) which is average of distances between quality, recommended by the strategy and best achievable performance on considered grid of hyperparameters. All qualities are normalized.
