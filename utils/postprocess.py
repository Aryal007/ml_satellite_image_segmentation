#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 21:49:04 2020

@author: mibook
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

class PostProcess:
    def __init__(self):
        self.image = None
        self.img_erode = None
        self.img_dilate = None
        
    def read_image(self, filename):
        image_np = np.load(filename)
        image_np = image_np.astype('uint8')
        self.image = image_np.astype('uint8')
        return self.image
        
    def view_image(self):
        cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
        imS = cv2.resize(self.image, (3000, 3000))
        cv2.imshow("Original image", imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def erosion(self, image, kernel_size = 3, iterations = 1):
        kernel = np.ones((kernel_size,kernel_size), np.uint8) 
        self.img_erode = cv2.erode(image, kernel, iterations=iterations) 
        return self.img_erode
        
    def dilation(self, image, kernel_size = 3, iterations = 1):
        kernel = np.ones((kernel_size,kernel_size), np.uint8) 
        self.img_dilate = cv2.dilate(image, kernel, iterations=iterations) 
        return self.img_dilate

    def view_result(self, images, titles = ["Original", "Erode", "Dilate"]):
        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        for i, col in enumerate(ax):
            col.title.set_text(titles[i])
            col.imshow(images[i])
        plt.show()

if __name__ == "__main__":
    filename = "../output_masks/5_band13.npy"
    
    pp = PostProcess()
    
    image = pp.read_image(filename)
    
    img_dilation= pp.dilation(image, kernel_size = 3, iterations=3)
    
    img_erosion= pp.erosion(image, kernel_size = 3, iterations=3)
    
    pp.view_result([image, img_erosion, img_dilation])