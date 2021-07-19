import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import datasets, layers, models
import os
#Reading image data
import glob as gb
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Input
import tensorflow as tf
from random import shuffle

from os import listdir
from PIL import Image as PImage

def loadImages(path):
    # return array of images

    imagesList = listdir(r'C:\Users\saksham saxena\Desktop\train')
    loadedImages = []
    for image in imagesList:
        img = PImage.open(path + image)
        loadedImages.append(img)

    return loadedImages


#to get all image names in train file
#to get all image names in train file
Normalimages = os.listdir(r"C:\Users\saksham saxena\Desktop\Training Data\NORMAL")
COVID19images = os.listdir(r"C:\Users\saksham saxena\Desktop\Training Data\COVID19")
TB = os.listdir(r"C:\Users\saksham saxena\Desktop\Training Data\TB")
Pneumothorax = os.listdir((r"C:\Users\saksham saxena\Desktop\Training Data\Pneumothorax"))
Pneumonia = os.listdir((r"C:\Users\saksham saxena\Desktop\Training Data\PNEUMONIA"))





## Image Data Generator


train_datagen = ImageDataGenerator(rescale = 1./255,
      shear_range=0.2,
      zoom_range=0.2,horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale=1. / 255)


# NOTE: YOU MUST USE A BATCH SIZE OF 10 (batch_size=10) FOR THE
# TRAIN GENERATOR.
train_generator =train_datagen.flow_from_directory(r"C:\Users\saksham saxena\Desktop\train",
     batch_size= 10,
     shuffle=shuffle,
     target_size=(150,150), class_mode= "categorical"

)

test_generator =test_datagen.flow_from_directory(r'C:\Users\saksham saxena\Desktop\test',
     batch_size= 50,
     shuffle=shuffle,
     target_size=(150, 150), class_mode= "categorical"

)





trainShape=train_generator.__getitem__(0)[0].shape
testShape=test_generator.__getitem__(0)[0].shape
#Shape of Data
print("Train Shape \n",trainShape)
print("Test Shape \n",testShape)




Labels={'COVID19':0,'NORMAL':1,'PNEUMONIA':2,'PNEUMOTHORAX':3,'TB':4}



def getCode(label):
    return Labels[label]


# convert code to label
def getLabel(n):
    for x, c in Labels.items():
        if n == c:
            return x


#Reading image data
import glob as gb
import cv2
# to resize the all image as same size

#to read all images from directory
def getData(Dir,sizeImage):
    X=[]
    y=[]
    for folder in  os.listdir(Dir) : #to get the file name
        files = gb.glob(pathname= str( Dir  +"/" +folder+ '//*.jpg' )) # to get the images
        for file in files:
                picture=cv2.imread(file) #  or plt.imread(file)
                imageArray=cv2.resize(picture,(sizeImage,sizeImage))
                X.append(list(imageArray))
                y.append(getCode(folder))
    X=np.array(X)
    y=np.array(y)
    return X,y




#get train data
X_train, Y_train = getData(r"C:\Users\saksham saxena\Desktop\train",150)
# get test data
X_test , Y_test = getData(r'C:\Users\saksham saxena\Desktop\test',150)



print("X_train Shape",X_train.shape)
print("X_test Shape",X_test.shape)

print("Y_train Shape", Y_train.shape)
print("Y_test Shape", Y_test.shape)






#### Model Compilation

model = models.Sequential()
model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(5,activation='softmax'))





opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.SpecificityAtSensitivity(0.5), keras.metrics.SensitivityAtSpecificity(0.5), 'accuracy'])
model.summary()
model.fit(train_generator,epochs=30,validation_data=test_generator)

import pandas as pd

loss_df = pd.DataFrame(model.history.history)
print(loss_df)

loss_df[['loss', 'val_loss']].plot()
plt.show()

loss_df[['recall', 'val_recall']].plot()
plt.show()

loss_df[['precision', 'val_precision']].plot()
plt.show()
loss_df[['specificity_at_sensitivity', 'val_specificity_at_sensitivity']].plot()
plt.show()
loss_df[['sensitivity_at_specificity', 'val_sensitivity_at_specificity']].plot()
plt.show()

img_path = (r"C:\Users\saksham saxena\Desktop\train\COVID19\1-s2.0-S0929664620300449-gr2_lrg-a.jpg")
img = image.load_img(img_path, target_size=(150,150))


plt.imshow(img)
plt.show()

img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array,axis=0)

img_preprocessed = preprocess_input(img_batch)

prediction = model.predict(img_preprocessed)
print(prediction)




