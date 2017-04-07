from wget import bar_adaptive
from numpy import delete
from pip._vendor.pyparsing import line
from _csv import reader
print('Starting CNN script',  flush=True)

import csv
import cv2
import os
import numpy as np
import sys
import sklearn
from random import shuffle,uniform

print('Libraries imported', flush=True)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D

model = Sequential()
model.add(MaxPooling2D((2, 2),input_shape=(160*5,320,3)))
# model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(16, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('relu'))

model.compile(loss='mse', optimizer='adam')

print('Model compiled!', flush = True)
    
samples = []
firstline = True

# import urllib
import wget
import zipfile

input_data_zipfile = os.path.join('.','input','data.zip')
additional_data_dir = os.path.join('.','additional_data')
additional_data_unzipped_dir = os.path.join('.','additional_data','IMGho')
additional_data_zipfile = os.path.join('.','additional_data','IMGho.zip')
database_dir = os.path.join('.','data')
database_dir_OK = os.path.isdir(database_dir)
inputDirDataOK = os.path.isfile(input_data_zipfile)
AdditionalDataDirOK = os.path.isdir(additional_data_unzipped_dir)
# additional_data_OK = os.path.isdir(additional_data_unzipped_dir)

training_file = os.path.join('.','data','driving_log.csv')
trainFileOK = os.path.isfile(training_file)

import shutil
if inputDirDataOK:
    if not database_dir_OK:
        os.mkdir('data')
    shutil.copy(input_data_zipfile, database_dir)

if not AdditionalDataDirOK:
    zip_ref2 = zipfile.ZipFile(additional_data_zipfile, 'r')
    zip_ref2.extractall(additional_data_dir)
        
if not (database_dir_OK):
    print('Downloading database...', flush = True)
    fileaddress = 'https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip'
    zipdir = 'output'
    print('starting wget to:' +'.'+os.path.sep+zipdir,  flush=True)
    file_name = wget.download(fileaddress, out = '.'+os.path.sep+zipdir  ,bar=bar_adaptive)
#     file_name = '.\\output\\data.zip'
    print('wget finished',  flush=True)
    path_to_zip_file = os.path.join(file_name)
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')

    zip_ref.extractall('.'+os.path.sep)
    zip_ref.close()
    filesOrDirs = os.listdir('.'+os.path.sep)
    for fileInDir in filesOrDirs:
        database_dir_OK = os.path.isdir(fileInDir)
        if database_dir_OK:
            print(fileInDir + '\t\t\t Is dir? Yes',  flush=True)
        else:
            print(fileInDir + '\t\t\t Is dir? No',  flush=True)
            
    print('Download done...', flush = True)
else:
    print('Files OK, download not required.', flush = True)

zero_added =0
zero_dropped =0
sub_sample = []
fileLines =[]
with open(os.path.join('.','driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        fileLines.append(line)
    fileLines.pop(0)
    
    for idx,line in enumerate(fileLines):
        if(idx >4):
            for i in range(0,5):
                sub_sample.append(fileLines[i])
            samples.append(sub_sample)
            sub_sample = []

#         print(line)    


print('Size of sample: '+ str(len(samples)))
print('Size of zeros: '+ str(zero_added))


# print(samples)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=200):
    num_samples = len(samples)
#     print('Sample size: ' +str(num_samples))
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
#             print(batch_samples[0])
            images = []
            angles = []
            for batch_sample in batch_samples:
                rand_x = uniform(0,1)
                correction = 0.1
                if rand_x < 0.20:
                    img_index = 1
                    appl_correnction = correction
                elif rand_x > 0.80:
                    img_index = 2
                    appl_correnction  = 0
                else:
                    img_index = 0
                    appl_correnction = -correction

                source_paths = []
                for i in range(0,5):
                    source_paths.append(batch_sample[i][img_index])

                paths_converted = []
                for idx, path in enumerate(source_paths):
                    split_path = path.split('/')
                    paths_converted.append(os.path.join('.',*split_path))
 
#                 print(paths_converted[0])
                image = cv2.imread(paths_converted[0], cv2.IMREAD_COLOR)
#                 print('Initial size :' + str(image.shape)) 
                for idx in range(1,5):
#                     print(paths_converted[idx])
                    image_tmp = cv2.imread(paths_converted[idx], cv2.IMREAD_COLOR)
#                     print('Image_tmp size :' + str(image_tmp.shape)) 
                    image_cat = np.concatenate((image,image_tmp), axis=0)
                    image = image_cat
#                     print('Image size :' + str(image.shape)) 
                
                center_angle = float(batch_sample[4][4])
                image = image/255.0 -0.5
                
                images.append(image)
                angles.append(center_angle + appl_correnction)

            X_train = np.array(images)
            y_train = np.array(angles)
#             print(y_train)
#             print(len((X_train, y_train)),  flush=True)
            yield sklearn.utils.shuffle(X_train, y_train)
            



train_generator = generator(train_samples, batch_size=250)
validation_generator = generator(validation_samples, batch_size=50)


model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
                     validation_data=validation_generator, nb_val_samples=len(validation_samples), 
                     nb_epoch=10)
        




image = [cv2.imread(fileLines[10][0]),cv2.imread(fileLines[11][0]),cv2.imread(fileLines[12][0]),cv2.imread(fileLines[13][0]),cv2.imread(fileLines[13][0])]
print('Test image shape')
print(image.shape)
image_array = np.asarray(image)
print('Prediction:')
print(model.predict(image_array, batch_size=1))

model.save('model.h5')
