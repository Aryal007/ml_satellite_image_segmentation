from utils.data import Data 
from utils.estimators import Classifier, Dataset

tiff_filename = "/home/ecology/landwaterclassifier/Data/Images/"
shp_filename = "/home/ecology/landwaterclassifier/Data/Labels/"
data = Data(tiff_filename, shp_filename)
tiff = data.read_tiff()
mask = data.get_mask()
X,y = data.get_Xy(tiff, mask, n_sample = 2000000)
classifier = Classifier(savepath = "./outputs")
X_train, X_test, y_train, y_test = data.train_test_split(X, y, save=False)
all_dataset = Dataset(X_train, X_test, y_train, y_test)
classifier.random_forest(trainX=all_dataset.trainX, trainY=all_dataset.trainY, testX=all_dataset.testX, testY=all_dataset.testY, grid_search=True, train=False, feature_importance=False)
