#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 22:38:01 2020

@author: mibook
"""
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon, box
from rasterio.features import rasterize
from shapely.ops import cascaded_union

from matplotlib import pyplot
import numpy as np
from sklearn.model_selection import train_test_split
import os, glob

np.random.seed(7)

class Data:
    def __init__(self, tiff, shp, classes = ["water", "land"], default_channel = (1,2,3), savepath = "./outputs"):  
        if self.check_file_or_not(tiff) and self.check_file_or_not(shp):
            self.isfile = True
        else:
            self.isfile = False
        if "Background" in classes:
            self.background = True
            classes.remove("Background")
        else:
            self.background = False
        self.tiff_filename = tiff
        self.shp_filename = shp
        self.classes = list(sorted(classes))
        self.default_channel = default_channel
        self.savepath = savepath
        if not os.path.exists(savepath):
            os.makedirs(savepath)
    
    def check_file_or_not(self, x):
        if os.path.isfile(x):
            return True
        elif os.path.isdir(x):
            return False
        else:
            raise ValueError("Not a valid file or directory")
        
    def get_constants(self):
        print("Tiff: "+str(self.tiff_filename))
        print("Shp: "+str(self.shp_filename))
        print("Classes: "+str(self.classes))
        print("Default channel: "+str(self.default_channel))
        print("Savepath: "+str(self.savepath))

    def check_crs(self, crs_a, crs_b, verbose = False):
        """
        Verify that two CRS objects Match
        :param crs_a: The first CRS to compare.
            :type crs_a: rasterio.crs
        :param crs_b: The second CRS to compare.
            :type crs_b: rasterio.crs
        :side-effects: Raises an error if the CRS's don't agree
        """
        if verbose:
            print("CRS 1: "+crs_a.to_string()+", CRS 2: "+crs_b.to_string())
        if rasterio.crs.CRS.from_string(crs_a.to_string()) != rasterio.crs.CRS.from_string(
                crs_b.to_string()
        ):
            raise ValueError("Coordinate reference systems do not agree")
    
    def read_tiff(self):
        """
        This function reads the tiff file given 
        filename and returns the rasterio object
        Parameters
        ----------
        filename : string
        Returns
        -------
        rasterio tiff object
    
        """
        if self.isfile:
            dataset = rasterio.open(self.tiff_filename)
            return dataset
        else:
            files = glob.glob(self.tiff_filename+"/*.tif")
            dataset = [rasterio.open(file) for file in files]
            return dataset
    
    def read_shp(self, column, drop):
        """
        This function reads the shp file given 
        filename and returns the geopandas object
        Parameters
        ----------
        filename : string
        Returns
        -------
        geopandas dataframe
    
        """
        if self.isfile:
            shapefile = gpd.read_file(self.shp_filename)
            shapefile = shapefile[shapefile[column] != drop]
            return shapefile
        else:
            files = glob.glob(self.tiff_filename+"/*.tif")
            filename = [os.path.basename(file) for file in files]
            shp_filename = [file.replace(".tif",".shp") for file in filename]
            shp_paths = [self.shp_filename+file for file in shp_filename]
            shapefiles = []
            for shp_path in shp_paths:
                shapefile = gpd.read_file(shp_path)
                shapefiles.append(shapefile[shapefile[column] != drop])
            return shapefiles
    
    def get_mask(self, column="Classname", drop=None):
        """
        This function reads the tiff filename and associated
        shp filename given and returns the numpy array mask
        of the labels
        Parameters
        ----------
        tiff_filename : string
        shp_filename : string
        Returns
        -------
        numpy array of shape (channels * width * height)
    
        """
        
        #Generate polygon
        def poly_from_coord(polygon, transform):
            """
            Get a transformed polygon
            https://lpsmlgeo.github.io/2019-09-22-binary_mask/
            """
            poly_pts = []
            poly = cascaded_union(polygon)
            for i in np.array(poly.exterior.coords):
                poly_pts.append(~transform * tuple(i)[:2]) # in case polygonz format
            return Polygon(poly_pts)
        
        # Clip shapefile
        def clip_shapefile(img_bounds, img_meta, shp):
            """
            Clip Shapefile Extents to Image Bounding Box
            :param img_bounds: The rectangular lat/long bounding box associated with a
              raster tiff.
            :param img_meta: The metadata field associated with a geotiff. Expected to
              contain transform (coordinate system), height, and width fields.
            :param shps: A list of K geopandas shapefiles, used to build the mask.
              Assumed to be in the same coordinate system as img_data.
            :return result: The same shapefiles as shps, but with polygons that don't
              overlap the img bounding box removed.
            """
            bbox = box(*img_bounds)
            bbox_poly = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=img_meta["crs"].data)
            return shp.loc[shp.intersects(bbox_poly["geometry"][0])]
        
        def add_background_mask(tiff, mask):
            _binary_channel = np.any(mask == 1, axis=2)
            np_tiff = tiff.read()
            np_tiff[np_tiff == -32767] = np.nan
            _nonnan = np.isnan(np.mean(np_tiff[:-2,:,:], axis=0))
            index = np.invert(_nonnan+_binary_channel)
            background = np.zeros((mask.shape[0], mask.shape[1]))
            background[index] = 1
            background = np.expand_dims(background, axis=2)
            mask = np.append(mask, background, axis = 2)
            return mask
        
        datasets = self.read_tiff()
        shapefiles = self.read_shp(column, drop)
        if self.isfile:
            dataset = datasets
            shapefile = shapefiles
            shapefile_crs = rasterio.crs.CRS.from_string(str(shapefile.crs))
            if shapefile_crs != dataset.meta["crs"]:
                shapefile = shapefile.to_crs(dataset.meta["crs"].data)
            self.check_crs(dataset.crs, shapefile.crs)
            shapefile = clip_shapefile(dataset.bounds, dataset.meta, shapefile)
            mask = np.zeros((dataset.height, dataset.width, len(self.classes)))
            for key, value in enumerate(self.classes):
                geom = shapefile[shapefile[column] == value]
                poly_shp = []
                im_size = (dataset.meta['height'], dataset.meta['width'])
                for num, row in geom.iterrows():
                    if row['geometry'].geom_type == 'Polygon':
                        poly_shp.append(poly_from_coord(row['geometry'], dataset.meta['transform']))
                    else:
                        for p in row['geometry']:
                            poly_shp.append(poly_from_coord(p, dataset.meta['transform']))
                try:
                    channel_mask = rasterize(shapes=poly_shp, out_shape=im_size)
                    mask[:,:,key] = channel_mask
                except Exception as e:
                    print(e)
                    print(value)
            if self.background:
                mask = add_background_mask(datasets, mask)
            return mask
        else:
            masks = []
            for (shapefile, dataset) in zip(shapefiles, datasets):
                shapefile_crs = rasterio.crs.CRS.from_string(str(shapefile.crs))
                if shapefile_crs != dataset.meta["crs"]:
                    shapefile = shapefile.to_crs(dataset.meta["crs"].data)
                self.check_crs(dataset.crs, shapefile.crs)
                shapefile = clip_shapefile(dataset.bounds, dataset.meta, shapefile)
                mask = np.zeros((dataset.height, dataset.width, len(self.classes)))
                for key, value in enumerate(self.classes):
                    geom = shapefile[shapefile[column] == value]
                    poly_shp = []
                    im_size = (dataset.meta['height'], dataset.meta['width'])
                    for num, row in geom.iterrows():
                        if row['geometry'].geom_type == 'Polygon':
                            poly_shp.append(poly_from_coord(row['geometry'], dataset.meta['transform']))
                        else:
                            for p in row['geometry']:
                                poly_shp.append(poly_from_coord(p, dataset.meta['transform']))
                    try:
                        channel_mask = rasterize(shapes=poly_shp, out_shape=im_size)
                        mask[:,:,key] = channel_mask
                    except Exception as e:
                        print(e)
                        print(value)
                if self.background:
                    mask = add_background_mask(mask)
                masks.append(mask)
            return masks
    
    def get_tiff_details(self, tiff):
        """
        This function accepts the rasterio tiff object
        and print out it's details
    
            Parameters
        ----------
        tiff : rasterio tiff object
        Returns
        -------
        None.
        """
        print("Filename: "+tiff.name)
        print("Bands: "+ str(tiff.count))
        print("Width (pixels): "+ str(tiff.width))
        print("Height (pixels): "+ str(tiff.height))
        print("CRS: "+ str(tiff.crs))
        print("Bounds: "+ str(tiff.bounds))
        
    def view_tiff(self, tiff, channel = None):
        """
        This function accepts rasterio tiff object, 
        the channel/channels to visualize and displays
        tiff image
        Parameters
        ----------
        tiff : rasterio tiff object
        Returns
        -------
        None.
    
        """
        if channel is None:
            channel = self.default_channel
        if isinstance(channel, int):
            array = tiff.read(channel)
            array[array == -32767] = np.nan
            pyplot.imshow(array, cmap='pink')
            pyplot.show()
        elif len(channel) == 3:
            array = tiff.read(channel)
            array[array == -32767] = np.nan
            array = np.transpose(array, (1,2,0))
            minimum = np.nan_to_num(array).min(axis=(0,1))
            maximum = np.nan_to_num(array).max(axis=(0,1))
            array = (array - minimum ) / (maximum -minimum)
            pyplot.imshow(array)
            pyplot.show()
            
    def view_mask(self, mask):
        """
        This function accepts mask numpy array and visualizes the labels
        Parameters
        ----------
        tiff : rasterio tiff object
        Returns
        -------
        None.
        """
        classes = self.classes.copy()
        if mask.ndim > 2:
            _mask = np.zeros((mask.shape[0],mask.shape[1]))
            if self.background:
                classes.append("Background")
            for key, value in enumerate(classes):
                _mask += mask[:,:,key]*(key+1)
            pyplot.imshow(_mask, cmap = 'tab10')
            pyplot.show()
        else:
            pyplot.imshow(mask, cmap = 'tab10')
            pyplot.show()
    
    def view_tiff_labels(self, tiff, mask):
        """
        This function accepts rasterio tiff object, shapefile, 
        the channel/channels to visualize, classes and displays
        visual of 3-band image and its corresponding mask
        Parameters
        ----------
        tiff_filename : rasterio tiff object
        mask = numpy array
        classes = classes for numpy array
        Returns
        -------
        None.
    
        """
        if len(self.classes) == 2:
            fig = pyplot.figure()
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            
            tiff = tiff.read(self.default_channel)
            tiff[tiff == -32767] = np.nan
            ax = fig.add_subplot(1,4,1)
            ax.imshow(np.transpose(tiff, (1,2,0)))
            ax.axis('off')
            ax.set_title('Tiff image')
            
            ax = fig.add_subplot(1,4,2)
            ax.imshow(mask[:,:,0]*1+mask[:,:,1]*2)
            ax.axis('off')
            ax.set_title("Mask, combined")
            
            ax = fig.add_subplot(1,4,3)
            ax.imshow(mask[:,:,0])
            ax.axis('off')
            ax.set_title("Mask, "+self.classes[0])
            
            ax = fig.add_subplot(1,4,4)
            ax.imshow(mask[:,:,1])
            ax.axis('off')
            ax.set_title("Mask, "+self.classes[1])
            
            pyplot.show()  
            
    def _getXy(self, tiff, mask, n_sample, save):
        """
        This function gets rasterio tiff object, numpy mask,
        number of samples for each class and returns X,y
        Parameters
        ----------
        tiff : TYPE
            DESCRIPTION.
        mask : TYPE
            DESCRIPTION.
        n_sample : TYPE, optional
            DESCRIPTION. The default is 1000000.
        save : TYPE, optional
            DESCRIPTION. The default is False.
        Returns
        -------
        X : numpy array
            x values (no of bands * n_sample*len(classes))
        y : numpy array
            y values (len(classes) * n_sample*len(classes))
        """
        np_tiff = tiff.read()
        np_tiff[np_tiff == -32767] = np.nan
        _nonnan = np.isnan(np.mean(np_tiff[:-2,:,:], axis=0))
        classes = self.classes.copy()
        if self.background:
            classes.append("Background")
        n_sample = np.amin(np.asarray([len(np.where((mask[:,:,i] == 1) & (_nonnan == False))[0]) for i in range(len(classes))]+[n_sample]))
        X = np.zeros((n_sample*len(classes), np_tiff.shape[0]))
        y = np.zeros((n_sample*len(classes), len(classes)))
        for key, value in enumerate(classes):
            _mask = (mask[:,:,key] == 1) * (_nonnan == False)
            _mask = _mask.flatten()
            np_tiff = np_tiff.reshape(np_tiff.shape[0],-1)
            _X = np_tiff[:,_mask == 1]
            index = np.random.permutation(_X.shape[1])
            _X = _X[:,index[:n_sample]]
            X[key*n_sample:(key+1)*n_sample,:] = _X.T
            y[key*n_sample:(key+1)*n_sample,key] = 1
        if save:
            np.save(self.savepath+"/X.npy",X)
            np.save(self.savepath+"/y.npy",y)
        return X, y
            
    def get_Xy(self, tiff, mask, n_sample = 200000, save=False, k_fold=False):
        if self.isfile:
            X, y = self._getXy(tiff, mask, n_sample, save)
            return X, y
        else:
            X_list, y_list = [], []
            for tiff, mask in zip(tiff, mask):
                X, y = self._getXy(tiff, mask, n_sample, save)
                if k_fold:
                    X_list.append(X)
                    y_list.append(y)
                else:
                    X_list.extend(X)
                    y_list.extend(y)
            return np.asarray(X_list), np.asarray(y_list)
    
    def train_test_split(self, X, y, test_size = 0.25, save = True):
        """
        Sklearn implementation of train_test_split
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                    test_size=test_size, random_state=42, shuffle=True)
        if save:
            np.save(self.savepath+"/X_train.npy",X_train)
            np.save(self.savepath+"/y_train.npy",y_train)
            np.save(self.savepath+"/X_test.npy",X_test)
            np.save(self.savepath+"/y_test.npy",y_test)
        return X_train, X_test, y_train, y_test
            
    def get_histogram(self, X, y, channel=0):
        """
        This function takes X, y, channel and plots the histogram for that 
        channel in the X for all classes in y
        Parameters
        ----------
        X : Numpy array
            DESCRIPTION.
        y : Numpy array
            DESCRIPTION.
        channel : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        classes = self.classes.copy()
        if self.background:
            classes.append("Background")
        X = X[:,channel].astype('int16')
        try:
            bins = np.linspace(np.amin(X), np.amax(X), np.amax(X)-np.amin(X))
        except:
            bins = np.linspace(0, 100, 1)
        pyplot.title("Channel "+str(channel))
        for key, value in enumerate(classes):
            _x = X[y[:,key] == 1]
            pyplot.hist(_x, bins, alpha=0.5, density = True, label=value, log=True)
        pyplot.legend(loc='upper right')
        pyplot.ylabel('Probability')
        pyplot.xlabel('Intensity')
        pyplot.show()

    def convert_to_tiff(self, tiff, image_np, filename="test.tiff", nan_value = -32767):
        output = np.zeros(image_np.shape)
        with rasterio.open(self.savepath+"/"+filename,
                'w',
                nbits = 1,
                driver = tiff.profile["driver"], 
                dtype = tiff.profile["dtype"], 
                nodata = tiff.profile["nodata"],
                width = tiff.profile["width"],
                height = tiff.profile["height"],
                count = 1,
                crs = tiff.profile["crs"],
                transform = tiff.profile["transform"],
                blockxsize = tiff.profile["blockxsize"],
                blockysize = tiff.profile["blockysize"],
                tiled = tiff.profile["tiled"],
                interleave = tiff.profile["interleave"]) as dst:
            for i in range(np.amax(image_np+1)):
                output[image_np == i] = i
            tiff_np = tiff.read()
            output[np.mean(tiff_np, axis=0) == nan_value] = np.amax(image_np+1)
            dst.write(output.astype(tiff.profile["dtype"]), 1)

