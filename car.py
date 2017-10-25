import classifier
import numpy as np

''' convertFeatures() '''
'''
This function converts all the categorical features into numerical features
'''
def convertFeatures( X ):
	newX = X
	XShape = X.shape

	for row in range( XShape[0] ):

		''' Fixing buying '''
		if newX[ row, 0 ] == 'low':
			newX[ row, 0 ] = 1
		elif newX[ row, 0 ] == 'med':
			newX[ row, 0 ] = 2
		elif newX[ row, 0 ] == 'high':
			newX[ row, 0 ] = 3
		elif newX[ row, 0 ] == 'vhigh':
			newX[ row, 0 ] = 4
		else:
			newX[ row, 0 ] = 5

		''' Fixing maint '''	
		if newX[ row, 1 ] == 'low':
			newX[ row, 1 ] = 1
		elif newX[ row, 1 ] == 'med':
			newX[ row, 1 ] = 2
		elif newX[ row, 1 ] == 'high':
			newX[ row, 1 ] = 3
		elif newX[ row, 1 ] == 'vhigh':
			newX[ row, 1 ] = 4
		else:
			newX[ row, 1 ] = 5

		''' Fixing door '''
		if newX[ row, 2 ] == "5more":
			newX[ row, 2 ] = 4
		elif newX[ row, 2 ] == '4':
			newX[ row, 2 ] = 3
		elif newX[ row, 2 ] == '3':
			newX[ row, 2 ] = 2
		elif newX[ row, 2 ] == '2':
			newX[ row, 2 ] = 1
		else:
			newX[ row, 2 ] = 5

		''' Fixing person '''
		if newX[ row, 3 ] == '2':
			newX[ row, 3 ] = 1
		elif newX[ row, 3 ] == '3':
			newX[ row, 3 ] = 2
		elif newX[ row, 3 ] == 'more':
			newX[ row, 3 ] = 3
		else:
			newX[ row, 3 ] = 4

		''' Fixing lug_boot '''
		if newX[ row, 4 ] == 'small':
			newX[ row, 4 ] = 1
		elif newX[ row, 4 ] == 'med':
			newX[ row, 4 ] = 2
		elif newX[ row, 4 ] == 'big':
			newX[ row, 4 ] = 3
		else:
			newX[ row, 4 ] = 4

		''' Fixing safety '''
		if newX[ row, 5 ] == 'low':
			newX[ row, 5 ] = 1
		elif newX[ row, 5 ] == 'med':
			newX[ row, 5 ] = 2
		elif newX[ row, 5 ] == 'high':
			newX[ row, 5 ] = 3
		else:
			newX[ row, 5 ] = 4

	return X

''' Fetching Data '''
my_list = classifier.readFile( 'car_evals.csv' )
X, Y = classifier.getLabel( my_list, 'unacc' )
X = convertFeatures( X )
X = classifier.One_Hot_Encoding( X )
print "Car Evaluation Dataset"

''' Splitting data '''
print "\nTraining Size = 80%\n"
Xtrain, Xtest, Ytrain, Ytest = classifier.splitData( X, Y, 0.4 )

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

''' Splitting data '''
print "\nTraining Size = 60%\n"
Xtrain, Xtest, Ytrain, Ytest = classifier.splitData( X, Y, 0.4 )

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
