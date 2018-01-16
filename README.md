# MetaInit
This repository is dedicated to research on initialization of machine learning algorithms with hyperparameters which are learned from metafeatures describing the dataset.

The pipeline of the research is made as follows:

**./Datasets** folder contains 461 datasets which were downloaded from OpenML.org by filtering with conditions:
- <20000 instances
- <500 features
- 2 classes

Note: It is important that .arff files downloaded from OpenML.org contain a short description of each feature column. In each file it is said whether a column reflects numerical or categorical feature. This property is used in the code of the project.

Now we have to calculate a metafeature representation of each dataset. This can be made by applying function from **Get_descriptive_vector.py** at each dataset file. After that we can dump the matrix of all metafeature vectors to **A2.p** (this is already done actually).

Note: Not all files are succeded in calculating metafeature representation by different reasons. That's why in code we process all files in alphabetical order except for files with indices [91, 117, 137, 144, 328, 414, 423, 279, 129, 451].  


