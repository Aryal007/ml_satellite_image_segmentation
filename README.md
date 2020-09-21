# Satellite/Airborne image segmentation 

The primary purpose of this repository is to generate segmentation mask on overhead imagery. The example contain the classification of land/water on 35cm spatial resolution NOAA overhead images. It can also further be extended to 30m spatial resolution Landsat-7 images for glacier mapping on the HKH region.

To run the code on your machine:

1. Clone the repo: git clone https://github.com/Aryal007/ml_satellite_image_segmentation.git
2. Install requirements: pip3 install -r requirements.txt
3. The classes for data processing and learning algorithms are defined in utils/data.py and utils/estimators.py respectively. 
4. The ipython notebooks contain examples for training different images. The data is assumed to be in the folders ./Data/Images and ./Data/Labels for images and its corresponding labels.
5. The files should be run in the order:
    a. generate_data.py: To process the tiff images and corresponding shp labels and generate numpy arrays for training and testing data.
    b. grid_search.py: To find the optimal set of hyperparameters for learning algorithms. _Pass parameter train=True, grid_search=False to train using the optimal hyperparameters_
    c. get_predictions.py: To predict the labels using trained algorithm on input images.

## Land/Water Classification, Methodology: 

The images are provided by NOAA overhead images and the labels are generated manually using ArcGIS. The aim is to generate segmentation mask for the images given labels on some pixels of the image. The experimental results are presented in the notebook: https://github.com/Aryal007/ml_satellite_image_segmentation/blob/master/example_noaa_airborne.ipynb

_Original Image_
(./images/image.png)

_User Labels_
(./images/mask.png)

_Predicted Segmentation Mask_
(./images/predicted.png)
