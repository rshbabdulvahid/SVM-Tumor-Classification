# SVM-Tumor-Classification

The intent of this project was to test the effect of regularization on four different support-vector machine learning models: a regularized SVM w/ linear kernel, a
regularized SVM w/ Gaussian kernel, and unregularized versions of both. These models all trained on the same data set which described physical qualities of breast
tumors across a feature set of 30 key values. Each example is classified as either 'malignant' (1) or 'benign' (0) based on this feature set. The train-test split for
this data set is 75% for training and 25% for the hold-out dataset: there are a total of 570 examples. 


All SVM models were initalized within the sci-kit learn module "sklearn.svm". Other notable modules used include pandas for CSV parsing, sklearn modules for train-test-split
and learning_curves, and the sklearn module preprocessing, used to import StandardScaler for mean normalization. Finally, pyplot was used to plot learning curves.


The unregularized versions of both SVM models were initialized with no regularization whatsoever: the values for C (inverse of regularization parameter) and gamma (for 
gaussian model) are the default values initalized by sklearn. For the regularized version, GridSearchCV was used to find the optimal values for C and gamma (tuned around 
accuracy). Ultimately, the regularized gaussian performed best, with an accuracy score of 0.976. The accuracy scores for the unregularized linear SVM, regularized linear SVM,
and the unregularized Gaussian SVM are 0.964, 0.969, and 0.964 respectively. For both linear and gaussian, regularization resulted in better performance on the hold-out
data set.


In order to more visually see the effect on variance that regularization has on a model, learning curves were plotted. These learning curves plot cross validation scores and
training data scores (accuracy) against the number of training examples used for that particular data point: the spread of number of training examples used for data points are
{5, 10, 50, 100, 200, 250, 319} (319 represents 75% of the input data being used to train the model and 25% being used for cross-validation). 
Each learning curve is plotted using the pyplot package in python. As a final point, the StratifiedKFold module from sklearn was used to create each cross-validation data set while
plotting the learning curves. 


All the learning curves are shown below. It should be noted that the accuracy scores between the training set
and the cross validation set converges significantly tighter over increasing training examples for the regularized linear SVM when compared with the unregularized linear SVM.
For the Gaussian SVM, the difference in convergence is less significant and harder to see visually in the learning curve, though it's worth reiterating that the regularization
did indeed have a beneficial impact on accuracy score.



![Alt text](https://github.com/rshbabdulvahid/SVM-Tumor-Classification/blob/master/Unregularized_Lin.PNG)

![Alt text](https://github.com/rshbabdulvahid/SVM-Tumor-Classification/blob/master/Regularized_Lin.PNG)

![Alt text](https://github.com/rshbabdulvahid/SVM-Tumor-Classification/blob/master/Unregularized_Gauss.PNG)

![Alt text](https://github.com/rshbabdulvahid/SVM-Tumor-Classification/blob/master/Regularized_Gauss.PNG)
