from wget import bar_adaptive
from numpy import delete
print('Starting CNN script',  flush=True)


import csv
import cv2
import os
import numpy as np
import sys

print('Libraries imported', flush=True)


def kerasCallBack():
    sys.stdout.flush()

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D

model = Sequential()
# model.add(MaxPooling2D((2, 2),input_shape=(160*3,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
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
    
lines = []
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
AdditionalDataDirOK = os.path.isfile(additional_data_unzipped_dir)
additional_data_OK = os.path.isdir(additional_data_unzipped_dir)

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

with open(os.path.join('.','driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
images_left = []
images_right = []
measurements = []
import random
import gc

lines.pop(0)
random.shuffle(lines)

for i in range(0,10):
    print('Era: {}'.format(i))
    for idx,line in enumerate(lines):
        if idx!=0 and (idx%3000 == 0) or (idx == (len(lines)-1)):
            #size(None,160,320,3)
            x_train = np.array(images)     
            y_train = np.array(measurements)
            model.fit(x_train, y_train,nb_epoch=1, batch_size=1500 ,validation_split=0.1,shuffle=False)
            x_train = x_train[::,::,::-1,...]
            y_train = -np.array(measurements)
            model.fit(x_train, y_train,nb_epoch=1, batch_size=1500 ,validation_split=0.1,shuffle=False)
            images = []
            measurements = []
            gc.collect()
            
        source_path_front = line[0]
        source_path_left = line[1]
        source_path_right = line[2]
        filename_front = source_path_front.split('/')[-1]
        filename_left = source_path_left.split('/')[-1]
        filename_right = source_path_right.split('/')[-1]

        file_dir = source_path_front.split('/')[0]+os.path.sep+source_path_front.split('/')[1]
        
        current_path_front = os.path.join('.',file_dir,filename_front)
        current_path_left = os.path.join('.', file_dir,filename_left)
        current_path_right = os.path.join('.',file_dir,filename_right)
        image_front = cv2.imread(current_path_front, cv2.IMREAD_COLOR)
        image_left = cv2.imread(current_path_left, cv2.IMREAD_COLOR)
        image_right = cv2.imread(current_path_right, cv2.IMREAD_COLOR)

        if image_front is None:
            print(current_path_front + ' was not found', flush = True)
            
        correction = 0.2
        
        rand_x = random.uniform(0,1)
        if rand_x < 0.2:
            image = image_left
            measurement = float(line[3]) + correction
        elif rand_x > 0.8:
            image = image_right
            measurement = float(line[3])
        else:
            image = image_front
            measurement = float(line[3]) - correction

        image = image/255.0 -0.5
        images.append(image)
        measurements.append(measurement)

model.save('model.h5')
