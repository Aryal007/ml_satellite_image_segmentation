#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:48:53 2020

@author: mibook
"""
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection._split import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
import pickle
import time 
import os
from pure_sklearn.map import convert_estimator
from matplotlib import pyplot
import math

class Utils():
    @staticmethod
    def count(x):
        from collections import Counter
        c = Counter(x)
        return c


class Dataset():
    def __init__(self, trainX, testX, trainY, testY, 
                 classes = ["water", "land"]):
        self.raw = None
        self.trainX = trainX
        self.trainY = np.argmax(trainY, axis = 1)
        self.testX = testX
        self.testY = np.argmax(testY, axis = 1)
        self.classes = classes

    def get_train_data(self):
        return [self.trainX, self.trainY]

    def get_test_data(self):
        return [self.testX, self.testY]

    def num_classes(self):
        return len(self.classes)

    def num_data(self):
        return len(self.trainY) + len(self.testY)

    def info(self):
        print("No. of classes: {}".format(self.num_classes()))
        print ("Class labels: {}".format(self.classes))
        print ("Total data samples: {}".format(self.num_data()))

        if self.trainY is not None:
            print("Train samples: {}".format(len(self.trainY)))
            trainStat = Utils.count(self.trainY)
            for k in trainStat.keys():
                print("\t {}:{} = {}".format(k, self.classes[k], trainStat.get(k, 0)))

        if self.testY is not None:
            print ("Test stats: {}".format(len(self.testY)))
            testStat = Utils.count(self.testY)
            for k in testStat.keys():
                print("\t {}:{} = {}".format(k, self.classes[k], testStat.get(k, 0)))

class Classifier():
    def __init__(self, savepath = "./outputs",
                 bands = ["Red","Green","Blue","NIR", "NDVI"]):
        self.savepath = savepath
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        self.bands = bands
    
    def grid_search(self, estimator, param_grid, features, targets):
        print("\nGrid search for algorithm:  {}".format(estimator))
        cv = StratifiedShuffleSplit(n_splits=3, test_size=0.33, random_state=42)
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, verbose=10, n_jobs=-1)
        grid.fit(features, targets)
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
        return grid

    def train_and_evaluate(self, estimator, trainX, trainY, testX, testY):
        start = time.time()        
        estimator.fit(trainX, trainY)
        elapsed_time = time.time()-start
        print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
        print("Accuracy on train Set: ")
        print(estimator.score(trainX, trainY))
        print("Accuracy on Test Set: ")
        print(estimator.score(testX, testY))
        outputs = estimator.predict(testX)
        print("Classification Report: ")
        print(metrics.classification_report(testY, outputs))
        print("Confusion Matrix: ")
        print(metrics.confusion_matrix(testY, outputs))
        # convert to pure python estimator
        # https://medium.com/building-ibotta/predict-with-sklearn-20x-faster-9f2803944446
        _estimator = convert_estimator(estimator)
        pickle.dump(_estimator, open(self.savepath+'/estimator.sav', 'wb'))
        return estimator, outputs

    def naive_bayes(self, trainX, trainY, testX, testY, grid_search=False, train=True, alpha = 0.001):
        print('\nMultinominal Naive Bayes')
        if grid_search:
            estimator = MultinomialNB(alpha=alpha, fit_prior=True)
            alpha_range = [0.001, 0.002, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.07, 0.08, 0.09, 0.1, 0.5, 1, 1.2,
                           1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 100, 500, 1000]

            param_grid = dict(alpha=alpha_range)
            grid = self.grid_search(estimator=estimator, param_grid=param_grid,
                                    features=trainX, targets=trainY)
            alpha = grid.best_params_['alpha']

        if train:
            estimator = MultinomialNB(alpha=alpha)
            clf = Pipeline([
                ('clf', estimator)
            ])
            self.train_and_evaluate(clf, trainX, trainY, testX, testY)

    def svm_linear(self, trainX, trainY, testX, testY, grid_search=False, train=True, c = 1.5, gamma = "auto"):
        print('\nSVM with Linear Kernel')
        if grid_search:
            estimator = SVC(kernel='linear', random_state=42, verbose=False, C=c, gamma=gamma)
            C_range = [0.01, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 100, 500, 1000]
            gamma_range = [0.001, 0.01, 0.1, 1, 2, 3, "auto"]
            param_grid = dict(gamma=gamma_range, C=C_range)
            grid = self.grid_search(estimator=estimator, param_grid=param_grid,
                                    features=trainX, targets=trainY)
            c = grid.best_params_['C']
            gamma = grid.best_params_['gamma']

        if train:
            estimator = SVC(kernel='linear', random_state=42, verbose=False, C=c, gamma=gamma)
            clf = Pipeline([
                ('clf', estimator)
            ])
            self.train_and_evaluate(clf, trainX, trainY, testX, testY)

    def svm_rbf(self, trainX, trainY, testX, testY, grid_search=False, train=True, c = 100, gamma = 0.01):
        print('\nSVM with RBF Kernel')
        if grid_search:
            estimator = SVC(kernel='rbf', random_state=42, verbose=False, C=c, gamma=gamma)
            C_range = [0.01, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 100, 500, 1000]
            gamma_range = [0.001, 0.01, 0.1, 1, 2, 3, "auto"]
            param_grid = dict(gamma=gamma_range, C=C_range)
            grid = self.grid_search(estimator=estimator, param_grid=param_grid,
                                    features=trainX, targets=trainY)
            c = grid.best_params_['C']
            gamma = grid.best_params_['gamma']

        if train:
            estimator = SVC(kernel='rbf', random_state=42, verbose=False, C=c, gamma=gamma)
            clf = Pipeline([
                ('clf', estimator)
            ])
            self.train_and_evaluate(clf, trainX, trainY, testX, testY)
            
    def random_forest(self, trainX, trainY, testX, testY, grid_search=False, train=True, n_estimators = 10, max_depth = 3, feature_importance = False, roc_curve = False):
        print('\nRandom Forest')
        
        if grid_search:
            estimator = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state=42) 
            n_estimators_range = [10, 50, 100, 150]
            max_depth_range = [3, 5, 7, 10, 20]
            param_grid = dict(n_estimators=n_estimators_range, max_depth=max_depth_range)
            grid = self.grid_search(estimator=estimator, param_grid=param_grid,
                                    features=trainX, targets=trainY)
            n_estimators = grid.best_params_['n_estimators']
            max_depth = grid.best_params_['max_depth']

        if train:
            estimator = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state=42) 
            clf = Pipeline([
                ('clf', estimator)
            ])
            estimator, outputs = self.train_and_evaluate(clf, trainX, trainY, testX, testY)
            
            if feature_importance:
                self.get_rf_feature_importance(estimator)
            if roc_curve:
                pass
            
            return outputs

    def mlp(self, trainX, trainY, testX, testY, grid_search=False, train=True):
        print('\nMLP Neural Network')
        solver = 'adam'
        alpha = 0.000001
        learning_rate = 'adaptive'
        learning_rate_init = 0.0025
        momentum = 0.9
        hidden_layer_sizes = (256,)
        max_iter = 1000
        early_stopping = True
        if grid_search:
            estimator = MLPClassifier(solver=solver, alpha=alpha, learning_rate=learning_rate,
                                      learning_rate_init=learning_rate_init, momentum=momentum,
                                      hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42,
                                      verbose=False, early_stopping=early_stopping)
            solver_range = ['adam']
            alpha_range = [1e-6, 1e-5, 0.00001, 0.0001, 0.0005, 0.001, 0.002, 0.01, 0.1, 0.5, 1, 1.5]
            learning_rate_range = ['constant', 'adaptive']
            max_iter_range = [200, 500, 1000]
            momentum_range = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
            early_stopping_range = [True, False]
            learning_rate_init_range = [0.0001, 0.001, 0.0025, 0.01, 0.1, 1]
            hidden_layer_sizes_range = [(100,), (100, 50), (128, 64), (256, 64), (256, 128, 64)]

            param_grid = dict(solver=solver_range, alpha=alpha_range, learning_rate=learning_rate_range,
                              learning_rate_init=learning_rate_init_range,
                              hidden_layer_sizes=hidden_layer_sizes_range, max_iter=max_iter_range,
                              momentum=momentum_range, early_stopping=early_stopping_range)
            grid = self.grid_search(estimator=estimator, param_grid=param_grid,
                                    features=trainX, targets=trainY)
            solver = grid.best_params_['solver']
            alpha = grid.best_params_['alpha']
            learning_rate = grid.best_params_['learning_rate']
            learning_rate_init = grid.best_params_['learning_rate_init']
            momentum = grid.best_params_['momentum']
            hidden_layer_sizes = grid.best_params_['hidden_layer_sizes']
            max_iter = grid.best_params_['max_iter']
            early_stopping = grid.best_params_[early_stopping]

        if train:
            estimator = MLPClassifier(solver=solver, alpha=alpha, learning_rate=learning_rate,
                                      learning_rate_init=learning_rate_init, momentum=momentum,
                                      hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42,
                                      verbose=False, early_stopping=early_stopping)
            clf = Pipeline([
                ('vect', self.feature_extractor.get_extractor()),
                ('clf', estimator)
            ])
            self.train_and_evaluate(clf, trainX, trainY, testX, testY)
    
    def get_labels(self, tiff, estimator, k=16):
        loaded_model = pickle.load(open(self.savepath+estimator, 'rb'))
        np_tiff = tiff.read()
        np_tiff = np_tiff.transpose(1,2,0)
        # Check if values == 0 or 256 or nan for everything but last 2 axis
        mask = (np.mean(np_tiff[:,:,:-2], axis=2) == (256 or 0)) + (np.isnan(np.mean(np_tiff[:,:,:-2], axis=2)))
        height = np_tiff.shape[0]
        width = np_tiff.shape[1]
        np_tiff = np_tiff.reshape(-1, np_tiff.shape[2]).astype('float64')
        sample_size = math.ceil(height*width/k)
        for i in range(k):
            print(f"\tBatch {i+1} of {k}")
            local_tiff = list(np_tiff[sample_size*i:sample_size*(i+1),:])
            output = loaded_model.predict(local_tiff)
            try:
                outputs.extend(output)
            except:
                outputs = output
        outputs = np.asarray(outputs)
        outputs = outputs.reshape(height, width)
        outputs = outputs+1
        outputs[mask] = 0
        return outputs
    
    def get_rf_feature_importance(self, estimator):
        """
        This function takes random forest estimator and shows the 
        feature importance scores
        Parameters
        ----------
        estimator : scipy save model
            DESCRIPTION.
        Returns
        -------
        None.

        """
        feat_importances = pd.Series(estimator._final_estimator.feature_importances_, index=self.bands)
        feat_importances = feat_importances.sort_values(ascending=True).tail(10)
        feat_importances.plot.barh()
        pyplot.tight_layout()
        pyplot.show()
        
    def get_pr_curve(self, estimator, X_test, y_test, channel=0):
        """
        This function takes random forest estimator and shows the 
        feature importance scores
        Parameters
        ----------
        estimator : scipy save model
            DESCRIPTION.
        Returns
        -------
        None.

        """
        pred_probs = estimator.predict_proba(X_test)
        pred_probs = pred_probs[:,channel]
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, pred_probs)
        # plot the precision-recall curves
        no_skill = len(y_test[y_test[:,channel]==1]) / len(y_test)
        pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()
