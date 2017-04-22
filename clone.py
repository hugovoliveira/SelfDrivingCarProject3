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


from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D

# model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))



model = Sequential()
model.add(Convolution2D(24, 5, 5, input_shape=((160-60-25)*1,320,3), subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(48, 3, 3, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
    
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2, verbose=1, mode='auto')
model.compile(loss='mae', optimizer='adam')

print('Model compiled!', flush = True)
    
samples = []
firstline = True

# import urllib
import wget
import zipfile
import shutil


input_data_zipfile = os.path.join('.','input','data.zip')
additional_data_dir = 'additional_data'
database_dir = os.path.join('.','data')
database_dir_OK = os.path.isdir(database_dir)
training_file = os.path.join('.','data','driving_log.csv')
trainFileOK = os.path.isfile(training_file)

additional_data_list = [ name for name in os.listdir(additional_data_dir) if (os.path.isdir(os.path.join(additional_data_dir, name)) and name[0:3]!='skp') ]

if not (database_dir_OK):
    print('Downloading database...', flush = True)
    fileaddress = 'https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip'
    zipdir = 'output'
    print('starting wget to:' +'.'+os.path.sep+zipdir,  flush=True)
    file_name = wget.download(fileaddress, out = '.'+os.path.sep+zipdir  ,bar=bar_adaptive)
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
    general_counter = 0
    next(reader)
    for line in reader:
        if (abs(float(line[3])) > 0.01):
            fileLines.append(line)
            general_counter = general_counter+1
        else:
            if (ten_counter == 0):
                fileLines.append(line) 
            ten_counter = ten_counter+1
            if ten_counter > 4:
                ten_counter =0
                

for additional_data_set in additional_data_list:
    print(os.path.join(additional_data_dir, additional_data_set,'driving_log.csv'))
#     print(additional_data_dir)
    additional_data_set_dir =os.path.join(additional_data_dir, additional_data_set)
    with open(os.path.join(additional_data_set_dir,'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        ten_counter = 0;
        next(reader)
        for line in reader:
            for i in range(0,3):
                line[i] = additional_data_set_dir + '//' + line[i]
                fileLines.append(line)
    
                
samples =[]
steer_abs_sum = 0
steer_sum = 0
correction = 0.15
recover_correction = 0.15
for line in fileLines:
#     steer_angle = -float(line[7])
#     samples.append([line[0], steer_angle, 'front camera - flip'])
#     steer_abs_sum = steer_abs_sum + abs(steer_angle)
#     steer_sum = steer_sum + steer_angle    
    for i in range(0,6):
        if  i == 0:
            steer_angle = float(line[3])
            if 'recovery right' in line[0]:
                steer_angle = steer_angle- recover_correction
            if 'recovery left' in line[0]:
                steer_angle = steer_angle+ recover_correction
            samples.append([line[0], steer_angle, 'front camera'])
            steer_abs_sum = steer_abs_sum + abs(steer_angle)  
            steer_sum = steer_sum + steer_angle                  
        elif i == 1:
            if 'recovery right' not in line[0]:
                steer_angle = float(line[3])+correction
                samples.append([line[1], steer_angle, 'left camera'])
                steer_abs_sum = steer_abs_sum + abs(steer_angle)
                steer_sum = steer_sum + steer_angle     

        elif i == 2:
            if 'recovery left' not in line[0]:            
                steer_angle = float(line[3])-correction
                samples.append([line[2], steer_angle, 'right camera'])
                steer_abs_sum = steer_abs_sum + abs(steer_angle)
                steer_sum = steer_sum + steer_angle

        elif i == 3:
            steer_angle = -float(line[3])
            samples.append([line[0], steer_angle, 'front flip'])
            steer_abs_sum = steer_abs_sum + abs(steer_angle)  
            steer_sum = steer_sum + steer_angle                  
        elif i == 4:
            if 'recovery' not in line[0]:
                steer_angle = -(float(line[3])+correction)
                samples.append([line[1], steer_angle, 'left flip'])
                steer_abs_sum = steer_abs_sum + abs(steer_angle)
                steer_sum = steer_sum + steer_angle     
        elif i == 5:
            if 'recovery' not in line[0]:
                steer_angle = -(float(line[3])-correction)
                samples.append([line[2], steer_angle, 'right flip'])
                steer_abs_sum = steer_abs_sum + abs(steer_angle)
                steer_sum = steer_sum + steer_angle
        
mean_steering = steer_sum/float(len(samples))
steer_abs_mean = steer_abs_sum/float(len(samples))

print('Mean steering angle: ' + str(steer_abs_mean))
print('Mean steering: ', mean_steering)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Number of initial images: {}'.format(len(train_samples)))


def generator(samples, batch_size=64):
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
                converted_path = os.path.join('.',*path_split)
#                 print(converted_path)

#                 print(converted_path,flush=True)
                image = cv2.imread(converted_path, cv2.IMREAD_COLOR)[60:135,:,:]
#                 print(batch_a_sample[2][-3:])
                if batch_a_sample[2][-4:] == 'flip':
                    image = cv2.flip(image,1)
#                     print('Flipped')                    

                timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                marked_image = image.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(marked_image, batch_a_sample[2] + 'Strng:' + str(angle),(20,60), font, 0.5,(255,255,255),1,cv2.LINE_AA)
#                 cv2.imwrite(timestamp +'.jpg', marked_image)            
                image = image/255.0 -0.5
                image_vec.append(image)
                steerinc_vec.append(angle)


            X_train = np.array(image_vec)
            y_train = np.array(steerinc_vec)
#             print(len(y_train))
#             print(y_train)
#             print(len((X_train, y_train)),  flush=True)
            yield sklearn.utils.shuffle(X_train, y_train)
#             yield (X_train, y_train)
            



train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=6)



model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
                     validation_data=validation_generator, nb_val_samples=len(validation_samples), 
                     nb_epoch=20, callbacks=[earlyStopping])
        


model.save('model.h5')

# 
# for j in range(500,2000):
#     image = cv2.imread(fileLines[j][0])[25:100,:,:]
#     image_array = np.asarray([image])
#     image_array = image_array/255.0 - 0.5
#     steering = model.predict(image_array, batch_size=1)
#     print('Prediction: ' +str(steering) + '. Actual: ' + fileLines[j][3])