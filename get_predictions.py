from utils.estimators import Classifier
from utils.data import Data
import numpy as np

tiff_loc = "./Data/Images/"
shp_loc = "./Data/Labels/"
all_data = Data(tiff_loc, shp_loc, classes = ["water", "land"])
all_tiff = all_data.read_tiff() 
classifier = Classifier()
for tiff in all_tiff:
    prediction = classifier.get_labels(tiff,"/estimator.sav")
    np.save("./outputs/"+str(tiff.name).split("/")[-1].replace(".tif",".npy"), prediction)
