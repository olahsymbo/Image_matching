# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#############################################################################
########################### Library Declaration #############################
#############################################################################

import numpy as np
import cv2
import os
import glob 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import accuracy_score
import shutil

##############################################################################
##############################################################################
################### Make function for patches generation######################
##############################################################################

def croppat(imagr,windowsize_r,windowsize_c):
    datanr = []
    vectnr = []
    for r in range(0,imagr.shape[0] - windowsize_r, windowsize_r):
            for c in range(0,imagr.shape[1] - windowsize_c, windowsize_c):
                window = imagr[r:r+windowsize_r,c:c+windowsize_c]
            datanr.append(window)
            vectnr.append(window.flatten())
    return datanr, vectnr

#d = 0
#image_list_a = []
#    data.append(img)
#    xy = str(files[(f1)])
#    image_list_a.append(im)
#   filename = filename = "/home/vcs/Documents/consultant/Aout/" + f1.split("/")[-1]    
# Crop out the window   name='%s%s.png' % str(f1)
#        window = croppat(image_list_a,windowsize_r,windowsize_c)

##############################################################################
##############################################################################
################### Read in Images from Directory A ##########################
##############################################################################

img_dir = "C:/Users/olahs/Documents/Python_scripts/consultant/A/" # Enter Directory of all images 
outpath = "C:/Users/olahs/Documents/Python_scripts/consultant/Aout/"
data_path = os.path.join(img_dir,'*g')


# Define the window size and image size for reshaping
windowsize_r = 50
windowsize_c = 50
imagesize_r = 1500
imagesize_c = 1500

files = glob.glob(data_path) 
#data1 = []
data1 = np.empty((0, windowsize_r * windowsize_r), dtype = int)
for f1 in files:
    image_list = cv2.imread(f1)
    image_list_a = cv2.resize(image_list, (imagesize_r, imagesize_c)) 
    grayn = cv2.cvtColor(image_list_a, cv2.COLOR_BGR2GRAY)
    
    datanr, vectnr = croppat(grayn,windowsize_r,windowsize_c)
    
    ngg = f1.split("/")[-1]
    f1n = ngg[0:2]
    shutil.rmtree(os.path.join(img_dir,f1n))
    os.makedirs(os.path.join(outpath,f1n))
    
    for i in range(len(datanr)):
        mh = datanr[i]
        filename=os.path.join(outpath,f1n,"%s.png" % i)
        cv2.imwrite(filename, mh)
#    filename=os.path.join('/home/vcs/Documents/consultant/Aout',"%s.png" % r)
#            filename = filename = "/home/vcs/Documents/consultant/Aout/" 
#            cv2.imwrite(filename, window)
#    data1.append(np.array(vectnr))
    vnn1 = np.array(vectnr)
    data1 = np.append(data1, vnn1, axis=0)
#    data1.append(np.array(vectnr))
##    data1 = np.concatenate(data1, np.array(vectnr))



##############################################################################
##############################################################################
################### Read in Images from Directory B ##########################
##############################################################################

img_dir = "C:/Users/olahs/Documents/Python_scripts/consultant/B/" # Enter Directory of all images 
outpath = "C:/Users/olahs/Documents/Python_scripts/consultant/Bout/"
data_path = os.path.join(img_dir,'*g')

#def croppat(image_list_a,windowsize_r,windowsize_c):
#    for r in range(0,image_list_a.shape[0] - windowsize_r, windowsize_r):
#        for c in range(0,image_list_a.shape[1] - windowsize_c, windowsize_c):
#            window = image_list_a[r:r+windowsize_r,c:c+windowsize_c]
#            return window

#d = 0
#image_list_a = []

files = glob.glob(data_path) 
#data2 = np.array([])
data2 = np.empty((0, windowsize_r * windowsize_r), dtype = int)
 
for f1 in files:
    image_list = cv2.imread(f1)
    image_list_a = cv2.resize(image_list, (imagesize_r, imagesize_c)) 
    grayn = cv2.cvtColor(image_list_a, cv2.COLOR_BGR2GRAY)
    
    datanr, vectnr = croppat(grayn,windowsize_r,windowsize_c)

    ngg = f1.split("/")[-1]
    f1n = ngg[0:2]
    shutil.rmtree(os.path.join(img_dir,f1n))
    os.makedirs(os.path.join(outpath,f1n))
    
    for i in range(len(datanr)):
        mh = datanr[i]
        filename=os.path.join(outpath,f1n,"%s.png" % i)
        cv2.imwrite(filename, mh)
#    filename=os.path.join('/home/vcs/Documents/consultant/Aout',"%s.png" % r)
#            filename = filename = "/home/vcs/Documents/consultant/Aout/" 
#            cv2.imwrite(filename, window)
#    data2.append(np.array(vectnr))
    vnn = np.array(vectnr)
    data2 = np.append(data2, vnn, axis=0)
#    data.append(img)
#    xy = str(files[(f1)])
#    image_list_a.append(im)
#   filename = filename = "/home/vcs/Documents/consultant/Aout/" + f1.split("/")[-1]    
# Crop out the window   name='%s%s.png' % str(f1)
#        window = croppat(image_list_a,windowsize_r,windowsize_c)
#    for r in range(0,gray.shape[0] - windowsize_r, windowsize_r):
#        for c in range(0,gray.shape[1] - windowsize_c, windowsize_c):
#            window = gray[r:r+windowsize_r,c:c+windowsize_c]
#            filename=os.path.join('/home/vcs/Documents/consultant/Bout',"%s.png" % r)
##            filename = filename = "/home/vcs/Documents/consultant/Aout/" 
#            cv2.imwrite(filename, window)
#            data2.append(window.flatten())


# Visualize image ande first patch
#imgplot = plt.imshow(grayn)
#plt.show()

#imgplot = plt.imshow(datanr[1], cmap='gray')
#plt.show()


##############################################################################
##############################################################################
################### Predict using SVM classifier  ############################
##############################################################################

At = [1] * len(data1)
Bt = [0] * len(data2)

M_data = np.concatenate((data1, data2))
y = np.concatenate((At, Bt))
X_train, X_test, y_train, y_test = train_test_split(M_data, y, test_size=0.35, random_state=42)


svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)  


y_pred = svclassifier.predict(X_test)  

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  

print(accuracy_score(y_test, y_pred)) 

