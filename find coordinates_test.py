# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:37:59 2018

@author: sivakumar
"""
import numpy as np
import os
from PIL import Image
from PIL import ImageOps
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import sys


from keras import backend as K
K.set_image_dim_ordering('th')

#load test image
testimage = Image.open(sys.argv[1])

#convert image into nump array and change shape to theano compatible form
testimgarray=np.array(testimage)


X_test=np.reshape(np.array(testimgarray),(1,3,326,490))
xtest = X_test.astype('float32')
xtest /= 255
"""
load trained model. Make sure that the "my_model1.h5" file is in the same folder as the script.
If you decide to use the trained model, make sure to put the my_model.h5 in the same folder as the script file 
and comment the first load _model and uncomment the second load model. This loads the trained model into model1.
"""
model1=load_model('my_model1.h5')
# model1=load_model('my_model.h5')



#predicting the coordinates and printing it.
ytest=model1.predict(xtest);
ynorm=[ytest[:,0]/326,ytest[:,1]/490]
print(str(ynorm[0])+" " +str(ynorm[1]))
