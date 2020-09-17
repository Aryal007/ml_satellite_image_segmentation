from utils.data import Data

tiff_filename = "./Data/Images/"
shp_filename = "./Data/Labels/"
data = Data(tiff_filename, shp_filename)
tiff = data.read_tiff()
mask = data.get_mask()
X,y = data.get_Xy(tiff, mask, n_sample = 2000000)
X_train, X_test, y_train, y_test = data.train_test_split(X, y)

