'''
Author: Ravi Patel
Date: 06/9/2017
'''

''' Importing python packages '''
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''----------------------------------------------- splitData() -------------------------------------------------'''
'''
Function Name: splitData()
Function Prototype: def splitData( X, Y, testSize )
Description: this function splits the input data into testing and training sets
Parameters:
	X - arg1 -- data containing all the features
	Y - arg2 -- data containing all the labels
	testSize - arg3 -- size of the testing data, in the range of (0,1) exclusive
Return Value: this function will return the following four datasets in this order
	Xtrain -- new training set containing all the features
	Xtest -- new testing set containing all the features
	Ytrain -- new training set containin all the labels of the corresponding training set
	Ytest -- new testing set containing all the labels of the corresponding testing set
'''
def splitData( X, Y, testSize ):
	Xtrain, Xtest, Ytrain, Ytest = train_test_split( X, Y, test_size = testSize, random_state = 42 )
	return Xtrain, Xtest, Ytrain, Ytest


''' redFile() '''
'''
This function reads from a given file and returns a list containing  the data from the given file
'''
def readFile( fileName ):
	with open( fileName, 'r' ) as f:
		reader = csv.reader( f )
		my_list = list( reader )
	my_list = np.asarray( my_list )
	return my_list

''' One_Hot_Encoding() '''
'''
This function performs one-hot-encoding on the input matrix and returns the new encoded matrix
'''
def One_Hot_Encoding( X ):
	enc = OneHotEncoder()
	enc.fit( X )
	newX = enc.transform( X ).toarray()
	return newX

''' getLabel() '''
'''
This function extracts the labels from the original list and modifies the features and returns the
updated features as well as the labels for each feature vector
'''
def getLabel( orig_list, class1 ):
	list_shape = orig_list.shape
	Y = np.zeros( list_shape[0] )
	for row in range( list_shape[ 0 ] ):
		for col in orig_list[ row, : ]:
			if col == class1: 
				Y[ row ] = 1
			else:
				Y[ row ] = -1

	X = np.delete( orig_list, -1, 1 )
	Y = Y.astype( int )
	return X, Y	


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''----------------------------------------- decision_tree_fit() -----------------------------------------------'''
'''
Function Name: decision_tree_train()
Function Prototype: def decision_tree_fit( Xtrain, Ytrain, depth = None )
Description: this function fit a decision tree model with a training set
Parameters:
	Xtrain - arg1 -- training dataset containing all the features
	Ytrain - arg2 -- training dataset containing all the labels 
	depth - opt agr3 -- depth of the decision tree (default = None )
	view - opt arg4 -- boolean value specifying whether to view the classifier of not (default = False )
	fileName - opt arg5 -- file name to be used for creating visualization
				must be ".dot" filename (default = "tree.dot" )
Return Value: object containing the fitted model of the training set
'''
def decision_tree_train( Xtrain, Ytrain, depth = None, view = False, fileName = "tree.dot" ):
	clf = tree.DecisionTreeClassifier( max_depth = depth )
	clf = clf.fit( Xtrain, Ytrain )
	if view == True:
		visualize_data( clf, fileName )
	return clf

'''---------------------------------------- decision_tree_predict() --------------------------------------------'''
'''
Function Name: decision_tree_classify()
Function Prototype: def decision_tree_predict( Xtest, classifier )
Description: this function predicts on a testing set using a decision tree classifier
Parameters:
	Xtest - arg1 -- testing dataset containing all the features
	classifier - arg2 -- classificating object that is returned from decision_tree_fit()
Return Value: object containing the predicted result of the classifier
'''
def decision_tree_classify( classifier, Xtest ):
	predict = classifier.predict( Xtest )
	return predict

'''---------------------------------------- decision_tree_score() ----------------------------------------------'''
'''
Function Name: decision_tree_score( Ytest, predict )
Function Prototype: def decision_tree_score( Ytest, predict )
Description: this function calculates the accuracy of the testing set based on the Decision Tree classifier.
Parameters:
	Ytest - arg1 -- testing dataset containing all the labels
	predict - arg2 -- prediction object that is returned from decision_tree_predict()
Return Value: classification error of the Decision Tree Classifier
'''
def decision_tree_score( predict, Ytest ):
	misClassified = 0
	#predict = predict.reshape( [ -1, 1 ] )
	#Ytest = Ytest.reshape( [ -1, 1 ] )
	result = predict + Ytest
	numOfTestData = float( result.size )
	for i in result:
		if i == 0:
			misClassified = misClassified + 1
	error = misClassified / numOfTestData
	return error

'''-------------------------------------------- decision_tree() ------------------------------------------------'''
'''
Function Name: decision_tree()
Function Prototype: def decision_tree( Xtrain, Ytrain, Xtest, Ytest, depth = None )
Description: this function runs decision_tree_fit(), decision_tree_predict() and decision_tree_score() functions
		in that order
Parameters:
	Xtrain - arg1 -- training dataset containing all the features
	Ytrain - arg2 -- training dataset containing all the labels
	Xtest - agr3 -- testing dataset containing all the features
	Ytest - arg4 -- testing dataset containing all the labels
	depth - opt arg5 -- depth of the Decsion Tree (Default = None )
	view - opt arg4 -- boolean value specifying whether to view the classifier of not (default = False )
	fileName - opt arg5 -- file name to be used for creating visualization
				must be ".dot" filename (default = "tree.dot" )
Return Value: classification error of the Decision Tree classifier
'''
def decision_tree( Xtrain, Ytrain, Xtest, Ytest, depth = None, view = False, fileName = "tree.dot" ):
	clf = decision_tree_train( Xtrain, Ytrain, depth, view, fileName )
	predict = decision_tree_classify( clf, Xtest )
	error = decision_tree_score( predict, Ytest )
	return error

'''--------------------------------------- K_Fold_crossValidation_Decision_Tree() --------------------------------------------'''
'''
Function Name: K_Fold_crossValidation_Decision_Tree_Census()
Function Prototype: def K_Fold_crossValidation( Xtrain, Ytrain, num_folds = 5 )
Description: this function performs K Fold Cross-Validation on Decision Classifier to find th optimal depth of the
		classifier
Parameters:
	Xtrain - arg1 -- training set containing the features
	Ytrain - arg2 -- training set containing the labels
	num_folds - opt arg3 -- number of folds to make (Default = 5 )
Return Value: this function returns the follwing values in this order
	validation_err -- validation error corresponding to the optimal depth
	train_err -- training error corresponding to the optimal depth
	optimal_depth -- optimal depth for the Decision Tree classifier
'''
def K_Fold_crossValidation_Decision_Tree( Xtrain, Ytrain, num_folds = 5 ):
	err =  -1 * np.ones( [1, num_folds] )
	err_train = -1 * np.ones( [1, num_folds] )
	validation_err = 9999999.0
	train_err = 9999999.0
	optimal_depth = 0
	Xtrain_shape = Xtrain.shape
	
	for j in range( 1,  Xtrain_shape[0], 100 ):
		for i in range( num_folds ):
			lower = i * Xtrain_shape[0] / num_folds
			upper = lower + ( Xtrain_shape[0] / num_folds)
			
			newX_test = Xtrain[ lower : upper, : ]
			newY_test = Ytrain[ lower : upper ]
			
			newX_train = np.delete( Xtrain, range( lower, (upper + 1) ), 0 )
			newY_train = np.delete( Ytrain, range( lower, (upper + 1) ), 0 )
			
			err_train[0,i] = decision_tree( newX_train, newY_train, newX_train, newY_train, j )
			err[0,i] = decision_tree( newX_train, newY_train, newX_test, newY_test, j )
		
		avg = (np.sum( err ) / float( num_folds ) )
		if avg < validation_err:
			validation_err = avg
			optimal_depth = j
			train_err = (np.sum(err_train)/ float( num_folds ) )
	
	if optimal_depth != 1:
		for j in range( optimal_depth - 100, optimal_depth + 100, 10 ):
			for i in range( num_folds ):
				lower = i * Xtrain_shape[0] / num_folds
				upper = lower + ( Xtrain_shape[0] / num_folds)
			
				newX_test = Xtrain[ lower : upper, : ]
				newY_test = Ytrain[ lower : upper ]
				
				newX_train = np.delete( Xtrain, range( lower, (upper + 1) ), 0 )
				newY_train = np.delete( Ytrain, range( lower, (upper + 1) ), 0 )
			
				err_train[0,i] = decision_tree( newX_train, newY_train, newX_train, newY_train, j )
				err[0,i] = decision_tree( newX_train, newY_train, newX_test, newY_test, j )
		
			avg = (np.sum( err ) / float( num_folds ) )
			if avg < validation_err:
				validation_err = avg
				optimal_depth = j
				train_err = (np.sum(err_train)/ float( num_folds ) )
	
	return validation_err, train_err, optimal_depth


'''------------------------------------------- visualize_data()  -----------------------------------------------'''
'''
Function Name: visualize_data()
Function Prototype: def visualize_data( clf, fileName = "tree.dot" )
Description: this function created a file containing a visualization of the Decision Tree classifier in the 
	directory containing the source file
Parameters:
	clf - arg1 -- classifier object with an already fitted training data
	fileName - opt arg2 -- filename to store the visualization
Return Value: none
'''
def visualize_data( clf, fileName = "tree.dot" ):
	print "Creating visualization..."
	tree.export_graphviz( clf, out_file = fileName, filled = True, class_names = True )
	print "Look for a file called %s in the directory containing the source file" %(fileName)

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" '''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''---------------------------------------- KNN() ----------------------------------------- '''
'''
Function Name: KNN()
Function Prototype: def KNN( Xtrain, Ytrain, Xtest, Ytest, K )
Description: this function runs K Nearest Neighbor classifier by calling the KNNTrain(),
	KNNClassify() and KNNTest() function one after the other
Parameter:
	Xtrain - arg1 -- data set containing all the features of the training set
	Ytrain - arg2 -- data set containing all the labels of the training set
	Xtest - arg3 -- data set containing all the features of the testing set
	Ytest - arg4 -- data set containing all the labels of the testing set
	K - arg4 -- number of nearest neighbor to take into consideration
Return Vale: classification error rate to of the K Nearest Neighbor classifier	
'''
def KNN( Xtrain, Ytrain, Xtest, Ytest, K = 1 ):
	clf = KNeighborsClassifier( n_neighbors = K )
	clf.fit( Xtrain, Ytrain )
	accuracy = clf.score( Xtest, Ytest )
	return ( 1 - accuracy )

'''
K_Fold_Cross_Validation_KNN()
'''
def K_Fold_crossValidation_KNN( Xtrain, Ytrain, num_folds = 5 ):
	err =  -1 * np.ones( [1, num_folds] )
	err_train = -1 * np.ones( [1, num_folds] )
	validation_err = 9999999.0
	train_err = 9999999.0
	optimal_K = 0
	Xtrain_shape = Xtrain.shape
	
	for j in range( 1,  Xtrain_shape[0], 100 ):
		for i in range( num_folds ):
			lower = i * Xtrain_shape[0] / num_folds
			upper = lower + ( Xtrain_shape[0] / num_folds)
			
			newX_test = Xtrain[ lower : upper, : ]
			newY_test = Ytrain[ lower : upper ]
			
			newX_train = np.delete( Xtrain, range( lower, (upper + 1) ), 0 )
			newY_train = np.delete( Ytrain, range( lower, (upper + 1) ), 0 )
			
			newX_train_Shape = newX_train.shape
			if j >= newX_train_Shape[0]:
				break

			err_train[0,i] = KNN( newX_train, newY_train, newX_train, newY_train, j )
			err[0,i] = KNN( newX_train, newY_train, newX_test, newY_test, j )
		
		avg = (np.sum( err ) / float( num_folds ) )
		if avg < validation_err:
			validation_err = avg
			optimal_K = j
			train_err = (np.sum(err_train)/ float( num_folds ) )

	if optimal_K != 1:
		for j in range( optimal_K - 100, optimal_K + 100, 10 ):
			for i in range( num_folds ):
				lower = i * Xtrain_shape[0] / num_folds
				upper = lower + ( Xtrain_shape[0] / num_folds)
			
				newX_test = Xtrain[ lower : upper, : ]
				newY_test = Ytrain[ lower : upper ]
				
				newX_train = np.delete( Xtrain, range( lower, (upper + 1) ), 0 )
				newY_train = np.delete( Ytrain, range( lower, (upper + 1) ), 0 )
			
				newX_train_Shape = newX_train.shape
				if j >= newX_train_Shape[0]:
					break

				err_train[0,i] = KNN( newX_train, newY_train, newX_train, newY_train, j )
				err[0,i] = KNN( newX_train, newY_train, newX_test, newY_test, j )
		
			avg = (np.sum( err ) / float( num_folds ) )
			if avg < validation_err:
				validation_err = avg
				optimal_K = j
				train_err = (np.sum(err_train)/ float( num_folds ) )

	return validation_err, train_err, optimal_K

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
''' SVM_train() '''
'''
Kernels:linear
	polynomial degree 2, 3,...
	rbf, with width (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2)
C: factor of ten from 10^(-7) to 10^3
'''
def SVM_train( Xtrain, Ytrain, ker = 'linear', C_value = 1 ):
	clf = SVC( kernel = ker, C = C_value )
	clf.fit( Xtrain, Ytrain )
	return clf

''' SVM_classify() '''
def SVM_classify( clf, Xtest ):
	classification = clf.predict( Xtest )
	return classification

''' SVM_score() '''
def SVM_score( clf, Xtest, Ytest ):
	accuracy = clf.score( Xtest, Ytest )
	return ( 1 - accuracy )

''' SVM() '''
def SVM( Xtrain, Ytrain, Xtest, Ytest, ker = 'linear', C_value = 1 ):
	clf = SVM_train( Xtrain, Ytrain, ker, C_value )
	err = SVM_score( clf, Xtest, Ytest )
	return err
'''
K_Fold_Cross_Validation_SVM()
'''
def K_Fold_crossValidation_SVM( Xtrain, Ytrain, ker = 'linear', num_folds = 5 ):
	err =  -1 * np.ones( [1, num_folds] )
	err_train = -1 * np.ones( [1, num_folds] )
	validation_err = 9999999.0
	train_err = 9999999.0
	optimal_C = 0
	Xtrain_shape = Xtrain.shape
	arr = [ 0.01, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0 ]
	
	for j in arr:
		for i in range( num_folds ):
			lower = i * Xtrain_shape[0] / num_folds
			upper = lower + ( Xtrain_shape[0] / num_folds)
			
			newX_test = Xtrain[ lower : upper, : ]
			newY_test = Ytrain[ lower : upper ]
			
			newX_train = np.delete( Xtrain, range( lower, (upper + 1) ), 0 )
			newY_train = np.delete( Ytrain, range( lower, (upper + 1) ), 0 )
			
			err_train[0,i] = SVM( newX_train, newY_train, newX_train, newY_train, ker, j )
			err[0,i] = SVM( newX_train, newY_train, newX_test, newY_test, ker, j )
		
		avg = (np.sum( err ) / float( num_folds ) )
		if avg < validation_err:
			validation_err = avg
			optimal_C = j
			train_err = (np.sum(err_train)/ float( num_folds ) )
		print j	
	return validation_err, train_err, optimal_C

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
''' logistic_regression_train() '''
'''
C: inverse of the regularization (10^(-8) to 10^4, change by factor of 10 )
'''
def logistic_regression_train( Xtrain, Ytrain, C_value = 1 ):
	clf = LogisticRegression( C = C_value )
	clf.fit( Xtrain, Ytrain )
	return clf

''' logistic_regression_classify() '''
def logistic_regression_classify( clf, Xtest ):
	classification = clf.predict( Xtest )
	return classification

''' logistic_regression_score() '''
def logistic_regression_score( clf, Xtest, Ytest ):
	accuracy = clf.score( Xtest, Ytest )
	return ( 1 - accuracy )

''' logistic_regression() '''
def logistic_regression( Xtrain, Ytrain, Xtest, Ytest, C_value = 1 ):
	Xtrain = Xtrain.astype( int )
	Ytrain = Ytrain.astype( int )
	Xtest = Xtest.astype( int )
	Ytest = Ytest.astype( int )
	clf = logistic_regression_train( Xtrain, Ytrain, C_value )
	err = logistic_regression_score( clf, Xtest, Ytest )
	return err

'''
K_Fold_Cross_Validation_logistic_regression()
'''
def K_Fold_crossValidation_logistic_regression( Xtrain, Ytrain, num_folds = 5 ):
	err =  -1 * np.ones( [1, num_folds] )
	err_train = -1 * np.ones( [1, num_folds] )
	validation_err = 9999999.0
	train_err = 9999999.0
	optimal_C = 0
	Xtrain_shape = Xtrain.shape
	#arr = [ 0.01, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0 ]	
	arr = np.linspace( 0.01, 10, 200 )	

	for j in arr:
		for i in range( num_folds ):
			lower = i * Xtrain_shape[0] / num_folds
			upper = lower + ( Xtrain_shape[0] / num_folds)
			
			newX_test = Xtrain[ lower : upper, : ]
			newY_test = Ytrain[ lower : upper ]
			
			newX_train = np.delete( Xtrain, range( lower, (upper + 1) ), 0 )
			newY_train = np.delete( Ytrain, range( lower, (upper + 1) ), 0 )

			newX_train = newX_train.astype( int )
			newY_train = newY_train.astype( int )
			newX_test = newX_test.astype( int )
			newY_test = newY_test.astype( int )
			
			err_train[0,i] = logistic_regression( newX_train, newY_train, newX_train, newY_train, j )
			err[0,i] = logistic_regression( newX_train, newY_train, newX_test, newY_test, j )
		
		avg = (np.sum( err ) / float( num_folds ) )
		if avg < validation_err:
			validation_err = avg
			optimal_C = j
			train_err = (np.sum(err_train)/ float( num_folds ) )
	return validation_err, train_err, optimal_C

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
''' random_forest_train() '''
def random_forest_train( Xtrain, Ytrain, num_of_trees = 10, depth = None ):
	clf = RandomForestClassifier( n_estimators = num_of_trees, max_depth = depth )
	clf.fit( Xtrain, Ytrain )
	return clf

''' random_forest_classify() '''
def random_forest_classify( clf, Xtest ):
	classification = clf.predict( Xtest )
	return classification

''' random_forest_score() '''
def random_forest_score( clf, Xtest, Ytest ):
	accuracy = clf.score( Xtest,Ytest )
	return ( 1 - accuracy )

''' random_forest() '''
def random_forest( Xtrain, Ytrain, Xtest, Ytest, num_of_trees = 10, depth = None ):
	clf = random_forest_train( Xtrain, Ytrain, num_of_trees, depth )
	err = random_forest_score( clf, Xtest, Ytest )
	return err

''' K_Fold_crossValidation_random_forest() '''
def K_Fold_crossValidation_Random_Forest( Xtrain, Ytrain, num_folds = 5 ):
	err =  -1 * np.ones( [1, num_folds] )
	err_train = -1 * np.ones( [1, num_folds] )
	validation_err = 9999999.0
	train_err = 9999999.0
	optimal_depth = 0
	optimale_num_of_trees = 0
	Xtrain_shape = Xtrain.shape
	arr = np.linspace( 1, 100, 100 )
	arr = arr.astype( int )
	depth = np.linspace( 1, 100, 100 )
	
	#for j in range( 1,  Xtrain_shape[0], 100 ):
	for j in arr:
		for k in depth:
			for i in range( num_folds ):
				lower = i * Xtrain_shape[0] / num_folds
				upper = lower + ( Xtrain_shape[0] / num_folds)
			
				newX_test = Xtrain[ lower : upper, : ]
				newY_test = Ytrain[ lower : upper ]
			
				newX_train = np.delete( Xtrain, range( lower, (upper + 1) ), 0 )
				newY_train = np.delete( Ytrain, range( lower, (upper + 1) ), 0 )
				
				err_train[0,i] = random_forest( newX_train, newY_train, newX_train, newY_train, num_of_trees = j, depth = k )
				err[0,i] = random_forest( newX_train, newY_train, newX_test, newY_test, num_of_trees = j, depth = k )
		
			avg = (np.sum( err ) / float( num_folds ) )
			if avg < validation_err:
				validation_err = avg
				optimal_depth = k
				optimal_num_of_trees = j
				train_err = (np.sum(err_train)/ float( num_folds ) )
		print j
	return validation_err, train_err, optimal_num_of_trees, optimal_depth

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
''' adaboost_train( ) '''
def adaboost_train( Xtrain, Ytrain, num_of_estimator = 50, learningRate = 1.0 ):
	clf = AdaBoostClassifier( n_estimators = num_of_estimator, learning_rate = learningRate )
	clf.fit( Xtrain, Ytrain )
	return clf

''' adaboost() '''
def adaboost( Xtrain, Ytrain, Xtest, Ytest, num_of_estimator = 50, learningRate = 1.0 ):
	clf = adaboost_train( Xtrain, Ytrain, num_of_estimator, learningRate )
	err = general_score( clf, Xtest, Ytest )
	return err

''' K_Fold_crossValidation_Adaboost() '''
def K_Fold_crossValidation_Adaboost( Xtrain, Ytrain, num_folds = 5 ):
	err =  -1 * np.ones( [1, num_folds] )
	err_train = -1 * np.ones( [1, num_folds] )
	validation_err = 9999999.0
	train_err = 9999999.0
	optimal_rate = 0
	Xtrain_shape = Xtrain.shape
	arr1 = np.linspace( 1, 100, 100 )
	arr1 = arr1.astype( int )
	rates = np.linspace( 0.01, 5, 10 )
	optimal_estimator = 0

	#for j in range( 1,  Xtrain_shape[0], 100 ):
	for j in arr1:
		for k in rates:
			for i in range( num_folds ):
				lower = i * Xtrain_shape[0] / num_folds
				upper = lower + ( Xtrain_shape[0] / num_folds)
			
				newX_test = Xtrain[ lower : upper, : ]
				newY_test = Ytrain[ lower : upper ]
			
				newX_train = np.delete( Xtrain, range( lower, (upper + 1) ), 0 )
				newY_train = np.delete( Ytrain, range( lower, (upper + 1) ), 0 )
			
				err_train[0,i] = adaboost( newX_train, newY_train, newX_train, newY_train, num_of_estimator = j, learningRate = k )
				err[0,i] = adaboost( newX_train, newY_train, newX_test, newY_test, num_of_estimator = j, learningRate = k )
			
			avg = (np.sum( err ) / float( num_folds ) )
			if avg < validation_err:
				validation_err = avg
				optimal_rate = k
				optimal_estimator = j
				train_err = (np.sum(err_train)/ float( num_folds ) )
		print j
	return validation_err, train_err, optimal_estimator, optimal_rate

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
''' MLPClassifier_train() '''
'''
Multi-layer Perceptron classifier
momentum >= 0, momentum <= 1
'''
def MLPClassifier_train( Xtrain, Ytrain, num_layers = (100,), moment = 0.9 ):
	clf = MLPClassifier( hidden_layer_sizes = num_layers, momentum = moment, solver = 'sgd' )
	clf.fit( Xtrain, Ytrain )
	return clf

''' MLP() '''
def MLP( Xtrain, Ytrain, Xtest, Ytest, num_layers = {100,}, moment = 0.9 ):
	Xtrain = Xtrain.astype( int )
	Xtest = Xtest.astype( int )
	Ytrain = Ytrain.astype( int )
	Ytest = Ytest.astype( int )
	clf = MLPClassifier_train( Xtrain, Ytrain, num_layers, moment )
	err = general_score( clf, Xtest, Ytest )
	return err

''' K_Fold_crossValidation_MLP() '''
def K_Fold_crossValidation_MLP( Xtrain, Ytrain, num_folds = 5 ):
	err =  -1 * np.ones( [1, num_folds] )
	err_train = -1 * np.ones( [1, num_folds] )
	validation_err = 9999999.0
	train_err = 9999999.0
	optimal_moment = 0
	Xtrain_shape = Xtrain.shape
	arr1 = np.linspace( 1, 200, 25 )
	arr1 = arr1.astype( int )
	momentum = np.linspace( 0, 1, 20 )
	optimal_layer = 0
		
	#for j in range( 1,  Xtrain_shape[0], 100 ):
	for j in arr1:
		for k in momentum:
			for i in range( num_folds ):
				lower = i * Xtrain_shape[0] / num_folds
				upper = lower + ( Xtrain_shape[0] / num_folds)
			
				newX_test = Xtrain[ lower : upper, : ]
				newY_test = Ytrain[ lower : upper ]
			
				newX_train = np.delete( Xtrain, range( lower, (upper + 1) ), 0 )
				newY_train = np.delete( Ytrain, range( lower, (upper + 1) ), 0 )
			
				err_train[0,i] = MLP( newX_train, newY_train, newX_train, newY_train, num_layers = {j,}, moment = k )
				err[0,i] = MLP( newX_train, newY_train, newX_test, newY_test, num_layers = {j,}, moment = k )
			
			avg = (np.sum( err ) / float( num_folds ) )
			if avg < validation_err:
				validation_err = avg
				optimal_moment = k
				optimal_layer = j
				train_err = (np.sum(err_train)/ float( num_folds ) )
		print j
	return validation_err, train_err, optimal_layer, optimal_moment



'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
''' general_classify() '''
def general_classify( classifier, Xtest ):
	classification = classifier.predict( Xtest )
	return classification

''' general_score() '''
def general_score( classifier, Xtest, Ytest ):
	accuracy = classifier.score( Xtest, Ytest )
	return ( 1 - accuracy )

