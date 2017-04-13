from wget import bar_adaptive
from datetime import datetime
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

# model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))

model = Sequential()
model.add(Convolution2D(6, 5, 5, input_shape=((160-60-25)*1,320,3)))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(16, 5, 5))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(40))
model.add(Activation('sigmoid'))
model.add(Dropout(0.7))
model.add(Dense(30))
model.add(Activation('sigmoid'))
model.add(Dropout(0.7))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mae', optimizer='adam')

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
    ten_counter = 0;
    next(reader)
    for line in reader:
        if (abs(float(line[3])) > 0.01):
            fileLines.append(line)
        else:
            if (ten_counter == 0):
                fileLines.append(line) 
            ten_counter = ten_counter+1
            if ten_counter > 9:
                ten_counter =0
                
samples =[]

correction = 0.1

for line in fileLines:
    for i in range(0,3):
        if i==0:
            samples.append([line[i], float(line[3])])
        elif i == 1:
            samples.append([line[i], float(line[3])+correction])
        else: #i==2
            samples.append([line[i], float(line[3])-correction])


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Number of initial images: {}'.format(len(train_samples)))


def generator(samples, batch_size=200):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            image_vec = []
            steerinc_vec = []
            for batch_a_sample in batch_samples:
                
                angle = batch_a_sample[1]
                path_split = batch_a_sample[0].split('/')
                converted_patch = os.path.join('.',*path_split)

                image = cv2.imread(converted_patch, cv2.IMREAD_COLOR)[60:135,:,:]

                timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
#                 cv2.imwrite('Strgt-ang {:.2f} - {}.jpg'.format(angle, timestamp), image)             
                image = image/255.0 -0.5
                image_vec.append(image)
                steerinc_vec.append(angle)
                image = cv2.flip(image,1)
                image_vec.append(image)
                steerinc_vec.append(-angle)

            X_train = np.array(image_vec)
            y_train = np.array(steerinc_vec)
#             print(y_train)
#             print(len((X_train, y_train)),  flush=True)
            yield sklearn.utils.shuffle(X_train, y_train)
#             yield (X_train, y_train)
            



train_generator = generator(train_samples, batch_size=100)
validation_generator = generator(validation_samples, batch_size=20)



model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*2, 
                     validation_data=validation_generator, nb_val_samples=len(validation_samples)*2, 
                     nb_epoch=8)
        


model.save('model.h5')


for j in range(500,550):
    image = cv2.imread(fileLines[j][0])[25:100,:,:]
    steering = model.predict([image], batch_size=1)
    print('Prediction: ' +str(steering))