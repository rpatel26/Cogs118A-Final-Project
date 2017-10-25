import classifier
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data = sio.loadmat( 'imageTrain.mat' )

Xtrain = data[ 'imageTrain' ]
tempXtrain = np.zeros( [ 5000, 784] )

for i in range( 5000 ):
	temp = Xtrain[ :, :, i ]
	tempXtrain[ i, : ] = temp.reshape( [ 1, 784 ] )
	

Ytrain = data[ 'labelTrain' ].reshape( [ -1, 1 ] )

Xtest = data[ 'imageTest' ]
tempXtest = np.zeros( [ 500, 784] )

for i in range( 500 ):
	temp = Xtest[ :, :, i]
	tempXtest[ i, : ] = temp.reshape( [ 1, 784 ] )


Ytest = data[ 'labelTest' ].reshape( [ -1, 1 ] )

Xtrain = tempXtrain
Xtest = tempXtest
Ytrain = np.ravel( Ytrain )
Ytest = np.ravel( Ytest )

''' convertLabels() '''
def convertLabels( Y, label ):
	Y_shape = Y.shape
	newLabels = np.zeros( Y_shape )
	
	for i in range( Y_shape[0] ):
		if Y[i] == label:
			newLabels[i] = 1
		else:
			newLabels[i] = -1

	newLabels = np.ravel( newLabels )
	return newLabels		


Ytest = convertLabels( Ytest, 1 )
Ytrain = convertLabels( Ytrain, 1 )

print "Handwritten Digit Dataset\n"
''' Decision Tree Classifier '''
val_err, train_err, max_depth = classifier.K_Fold_crossValidation_Decision_Tree( Xtrain, Ytrain )
test_err = classifier.decision_tree( Xtrain, Ytrain, Xtest, Ytest, depth = max_depth )
print "Decision Tree Classifier\n"
print "Validation Error = ", val_err
print "Training Error = ", train_err
print "Testing Error = ", test_err
print "Optimal Depth = ", max_depth
print "\n"

''' K Nearest Neighbor '''
val_err, train_err, opt_K = classifier.K_Fold_crossValidation_KNN( Xtrain, Ytrain )
test_err = classifier.KNN( Xtrain, Ytrain, Xtest, Ytest, K = opt_K )
print "K Nearest Neighbor Classifier\n"
print "Validation Error = ", val_err
print "Training Error = ", train_err
print "Testing Error = ", test_err
print "Optimal K = ", opt_K
print "\n"

''' SVM - linear kernel '''
val_err, train_err, opt_C = classifier.K_Fold_crossValidation_SVM( Xtrain, Ytrain, ker = 'linear' )
test_err = classifier.SVM( Xtrain, Ytrain, Xtest, Ytest, ker = 'linear', C_value = opt_C )
print "SVM - Linear Kernel Classifier\n"
print "Validation Error = ", val_err
print "Training Error = ", train_err
print "Testing Error = ", test_err
print "Optimal C = ", opt_C
print "\n"

''' SVM - RBF kernel '''
val_err, train_err, opt_C = classifier.K_Fold_crossValidation_SVM( Xtrain, Ytrain, ker = 'rbf' )
test_err = classifier.SVM( Xtrain, Ytrain, Xtest, Ytest, ker = 'rbf', C_value = opt_C )
print "SVM - RBF Kernel Classifier\n"
print "Validation Error = ", val_err
print "Training Error = ", train_err
print "Testing Error = ", test_err
print "Optimal C = ", opt_C
print "\n"


''' Logistic Regression '''
val_err, train_err, opt_C = classifier.K_Fold_crossValidation_logistic_regression( Xtrain, Ytrain )
test_err = classifier.logistic_regression( Xtrain, Ytrain, Xtest, Ytest, C_value = opt_C )
print "Logistic Regression classifier\n"
print "Validation Error = ", val_err
print "Training Error = ", train_err
print "Testing Error = ", test_err
print "Optimal C = ", opt_C
print "\n"


''' Random Forest '''
val_err, train_err, opt_num_of_trees, opt_depth = classifier.K_Fold_crossValidation_Random_Forest( Xtrain, Ytrain )
test_err = classifier.random_forest( Xtrain, Ytrain, Xtest, Ytest, num_of_trees = opt_num_of_trees, depth = opt_depth )
print "Random Forest Classifier\n"
print "Validation Error = ", val_err
print "Training Error = ", train_err
print "Testing Error = ", test_err
print "Optimal Number of Trees = ", opt_num_of_trees
print "Optimal Depth = ", opt_depth
print "\n"


''' Adaboost '''
val_err, train_err, opt_num_of_estimator, opt_rate = classifier.K_Fold_crossValidation_Adaboost( Xtrain, Ytrain )
test_err = classifier.adaboost( Xtrain, Ytrain, Xtest, Ytest, num_of_estimator = opt_num_of_estimator, learningRate = opt_rate )
print "Adaboost Classifier\n"
print "Validation Error = ", val_err
print "Training Error = ", train_err
print "Testing Error = ", test_err
print "Optimal Number of Estimator = ", opt_num_of_estimator
print "Optimal Learning Rate = ", opt_rate
print "\n"


''' Multi-Layer Perceptron '''
val_err, train_err, opt_layer, opt_moment = classifier.K_Fold_crossValidation_MLP( Xtrain, Ytrain )
test_err = classifier.MLP( Xtrain, Ytrain, Xtest, Ytest, num_layers = {opt_layer,}, moment = opt_moment )
print "Multi-Layer Perceptron Classifier\n"
print "Validation Error = ", val_err
print "Training Error = ", train_err
print "Testing Error = ", test_err
print "Optimal Number of Layers = ", opt_layer
print "Optimal Momentum = ", opt_moment
print "\n"

