"""
Created on Tue Feb 27 14:31:31 2018

@author: sivakumar

This script loads the labels.txt from the folder and loads the correspoding images. 
Data augmentation is performed on the images to get more data and a CNN is trainined using it. 
The training might take a very long time if not done on CPU. A trained netwrok is already stored in the submitted folder.
The testing program script can load the trained network and test on the images.
"""



import sys
import numpy as np
import os
from PIL import Image
from PIL import ImageOps


import glob
directory=sys.argv[1]
fname=directory+'labels.txt'

# loading the labels from the labels.txt
l=[]
l=np.genfromtxt(fname, dtype= str)

#loading the .jpg images in the same order as the labels. The images are stored as PIL image types
image_list = []
for item in l[:,0]:
     image_list.append(Image.open(directory+item))
    
# creating labels with just the coordinates in float type.
label=np.array([l[:,1],l[:,2]]);
label=label.transpose();
y = label.astype(np.float);

#%%
"""
functions for rotating and flipping images and finding accuracy after predicting 
The functions take in a list of images and corresponding labels to give out an augmented list of images 
with the actual images and augmented images along with their corresponding labels.
"""
def rotate(image_list,y):
    imager180=[]
    
    for item in image_list:
     
     imager180.append(item.rotate(180));
    
    labelnew=1-y;
    labeltot=np.concatenate((y,labelnew),axis=0);
    imagetot=image_list+imager180;
    return imagetot, labeltot


def flip(imagetot,labeltot):
    imagefliplr=[]
    for item in imagetot:
     imagefliplr.append(item.transpose(Image.FLIP_LEFT_RIGHT)); 
    labelflr=np.transpose(np.array([1-labeltot[:,0],labeltot[:,1]]));
    labeltot1=np.concatenate((labeltot,labelflr),axis=0);
    imagetot1=imagetot+imagefliplr;
    
    imageflipub=[]
    for item in imagetot1:
     imfub=item.transpose(Image.FLIP_TOP_BOTTOM)
     
     
     imageflipub.append(imfub);
    labelfub=np.transpose(np.array([labeltot1[:,0],1-labeltot1[:,1]]));
    labeltot2=np.concatenate((labeltot1,labelfub),axis=0);
    imagetot2=imagetot1+imageflipub;
    return imagetot2, labeltot2

def accuracy(rad):
    count=0;
    for item in rad:
      if item<=0.05:
        count=count+1;
    acc=count/len(rad)
    return acc
#%%
#data agumentation for increasing the training data.
imagetot,labeltot=rotate(image_list,y)
imagefull,labelfull=flip(imagetot,labeltot)
#%%
# converting the images into numpy arrays of required shape
data=[]
for item in imagefull:
    data.append(np.array(item))
# this shape is used for theano backend for keras (samples,color channels,height, width)    
data1=np.reshape(np.array(data),(1032,3,326,490))

#%%
# splitting data into training and vaidation.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data1, labelfull, test_size=0.2, random_state=0)

#data1=np.array(data)
#%%
# converting the normalised coordinates to pixel coordinates and normalising image values from 0 to 1.
trlab=np.transpose(np.array([y_train[:,1]*326,y_train[:,0]*490]))
xtrain = X_train.astype('float32')

xtrain /= 255

#%%

#building and training the keras model
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


from keras import backend as K
K.set_image_dim_ordering('th')



model = Sequential()
 
model.add(Convolution2D(50, 3, 3, input_shape=(3,326,490)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(20, 3, 3))
##model.add(MaxPooling2D(pool_size=(2, 2)))
##model.add(Convolution2D(10, 3, 3))


 
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))

model.add(Dense(2))
 
#  Compile model
model.compile(loss='mean_squared_error',
              optimizer='Adam',
              metrics=['accuracy'])
 
#  Fit model on training data
model.fit(xtrain, trlab, 
          batch_size=32, nb_epoch=100, verbose=1,shuffle='true')



"""
saving the trained model. the model gets saved in the ame folder as the script file.
"""
from keras.models import load_model

# Creates a HDF5 file 'my_model.h5'
model.save('my_model1.h5')

#%%
#make a prediction on training data
x=model.predict(xtrain)

#nomalise the predicted coordinates
xnorm=np.transpose(np.array([x[:,0]/326,x[:,1]/490]))

#%%
#error radius
rad_train=np.sqrt(np.power((xnorm[:,1]-trlabel[:,0]),2)+np.power((xnorm[:,0]-trlabel[:,1]),2))

#checking for training accuracy
train_acc=accuracy(rad_train)
#%%
"""
this part of the code gives out the validation accuracy.

"""





telab=np.transpose(np.array([y_test[:,1]*326,y_test[:,0]*490]))

xval = X_test.astype('float32')
xval /= 255
#X_test /= 255


xte=model.predict(xval)

xten=np.transpose(np.array([xte[:,0]/326,xte[:,1]/490]))


#error margin for validation
rad_test=np.sqrt(np.power((xten[:,1]-telabel[:,0]),2)+np.power((xten[:,0]-telabel[:,1]),2))


#validation accuracy
val_acc=accuracy(rad_test)

#%%




"""
the rest of the code is for using other methods such as linear regressiona and SVM
"""

#%%
"""
from sklearn import datasets, linear_model

Xtrain= np.reshape((xtrain),(825,479220))


from sklearn import decomposition
pca = decomposition.PCA(n_components=1000)
pca.fit(Xtrain)
Xtrain = pca.transform(Xtrain)

#%%
Yx=trlab[:,0]
Yy=trlab[:,1]


regrx = linear_model.LinearRegression()
regry = linear_model.LinearRegression()

regrx.fit(Xtrain, Yx)
regry.fit(Xtrain, Yy)
#%%
predx=regrx.predict(Xtrain)
predy=regry.predict(Xtrain)

xnorm=np.transpose(np.array([predy/326,predx/490]))
#%%
rad=np.sqrt(np.power((xnorm[:,1]-trlabel[:,0]),2)+np.power((xnorm[:,0]-trlabel[:,1]),2))


acc=accuracy(rad)


#%%
Xtest=np.reshape((xtest),(207,479220))

predx=regrx.predict(Xtest)
predy=regry.predict(Xtest)


xnorm=np.transpose(np.array([predy/326,predx/490]))

radtest=np.sqrt(np.power((xnorm[:,1]-telabel[:,0]),2)+np.power((xnorm[:,0]-telabel[:,1]),2))
acc=accuracy(radtest)

#%%
from sklearn.svm import SVR
Xtrain= np.reshape((xtrain),(825,479220))


from sklearn import decomposition
pca = decomposition.PCA(n_components=10)
pca.fit(Xtrain)
Xtrain = pca.transform(Xtrain)

Yx=trlab[:,0]
Yy=trlab[:,1]


regrx = SVR(kernel='linear', epsilon=0.05)
regry = SVR(kernel='linear', epsilon=0.05)

regrx.fit(Xtrain, Yx)
regry.fit(Xtrain, Yy)
#%%
predx=regrx.predict(Xtrain)
predy=regry.predict(Xtrain)

xnorm=np.transpose(np.array([predy/326,predx/490]))
rad=np.sqrt(np.power((xnorm[:,1]-trlabel[:,0]),2)+np.power((xnorm[:,0]-trlabel[:,1]),2))


acc=accuracy(rad)

#%%
Xtest=np.reshape((xtest),(207,479220))

predx=regrx.predict(Xtest)
predy=regry.predict(Xtest)


xnorm=np.transpose(np.array([predy/326,predx/490]))

radtest=np.sqrt(np.power((xnorm[:,1]-telabel[:,0]),2)+np.power((xnorm[:,0]-telabel[:,1]),2))
acc=accuracy(radtest)
"""