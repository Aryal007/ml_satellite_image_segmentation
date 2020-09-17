from utils.estimators import Classifier, Dataset
import numpy as np

savepath = "./outputs/"

X_train = np.load(savepath+"X_train.npy")
X_test = np.load(savepath+"X_test.npy")
y_train = np.load(savepath+"y_train.npy")
y_test = np.load(savepath+"y_test.npy")

classifier = Classifier(savepath = "./outputs")
all_dataset = Dataset(X_train, X_test, y_train, y_test)
classifier.random_forest(trainX=all_dataset.trainX, trainY=all_dataset.trainY, testX=all_dataset.testX, testY=all_dataset.testY, grid_search=True, train=False, feature_importance=False)
