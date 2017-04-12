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
model.add(Convolution2D(6, 5, 5),input_shape=((160-60-25)*5,320,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(16, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dropout(0.7))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dropout(0.7))
model.add(Dense(1))
model.add(Activation('relu'))

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
    for line in reader:
        fileLines.append(line)
    fileLines.pop(0)
    
    for idx,line in enumerate(fileLines):
        if(idx >4):
            for i in range(0,5):
                sub_sample.append(fileLines[idx-5+i])
#                 print(fileLines[i+idx])
            samples.append(sub_sample)
#             print(sub_sample[1][1] + '\n'+sub_sample[2][1])
#             print('')
#           [0, 1, 2, 3, 4]
            sub_sample = []


print('Size of sample: '+ str(len(samples)))
print('Size of zeros: '+ str(zero_added))


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=200):
    num_samples = len(samples)
    FirstSaved =0;
#     print('Sample size: ' +str(num_samples))
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
#             print(batch_samples[0])
            image_vec = []
            angles = []
            for batch_unit_of_five in batch_samples:
                
                rand_x = uniform(0,1)
                correction = 0.2
                if rand_x < 0.10:
                    img_index = 1
                    appl_correnction = correction
                elif rand_x > 0.90:
                    img_index = 2
                    appl_correnction  = 0
                else:
                    img_index = 0
                    appl_correnction = -correction

                source_paths = []
#                 print('Source paths')
                for i in range(0,5):
                    source_paths.append(batch_unit_of_five[i][img_index])
#                     print(batch_unit_of_five[i][img_index])
#                 print('\n')
                
                paths_converted = []
                for idx, path in enumerate(source_paths):
                    split_path = path.split('/')
#                     print(split_path)
                    paths_converted.append(os.path.join('.',*split_path))
 
#                 print(paths_converted[0])
                center_angle = (float(batch_unit_of_five[4][3])+float(batch_unit_of_five[3][3]))/2

                rand_x2 = uniform(0,1)
                if center_angle ==0 and rand_x2>0.03 and(rand_x>0.1 and rand_x<0.9):
                    print('Deleted sample')
                    continue
                
                expanded_image = np.ndarray([0,320,3])
                            
#                 print('Initial size :' + str(image.shape)) 
                for idx in range(5,0,-1):
                    new_image = cv2.imread(paths_converted[idx-1], cv2.IMREAD_COLOR)[60:135,:,:]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(new_image,paths_converted[idx-1],(0,30), font, 0.3,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(new_image,'idx: {}/strng: {}/rand_x:{:.2f}'.format(idx-1, batch_unit_of_five[idx-1][3], rand_x),(0,60), font, 0.3,(255,255,255),1,cv2.LINE_AA)
                    expanded_image = np.concatenate((expanded_image, new_image),0)
                    if FirstSaved == 0:
                        print(paths_converted[idx-1])
                    
#                     print('Image size :' + str(image.shape)) 
                
                
                expanded_image = expanded_image/255.0 -0.5
                if FirstSaved == 0:
                    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                    cv2.imwrite('{}.jpg'.format(timestamp), (expanded_image+0.5)*int(255))
                    FirstSaved =1
                
                image_vec.append(expanded_image)
                angles.append(center_angle + appl_correnction)

            X_train = np.array(image_vec)
            y_train = np.array(angles)
#             print(y_train)
#             print(len((X_train, y_train)),  flush=True)
            yield sklearn.utils.shuffle(X_train, y_train)
#             yield (X_train, y_train)
            



train_generator = generator(train_samples, batch_size=250)
validation_generator = generator(validation_samples, batch_size=50)



model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
                     validation_data=validation_generator, nb_val_samples=len(validation_samples), 
                     nb_epoch=8)
        


model.save('model.h5')


for j in range(500,550):
    image = cv2.imread(fileLines[j][0])[25:100,:,:]
    image = image/255.0 -0.5
    for i in range(j-4,j):
        image = np.concatenate((image, cv2.imread(fileLines[10+i][0])[25:100,:,:]),0)
        cv2.putText(image,fileLines[j][0],(0,30), font, 0.3,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(image,'strng: {}'.format(fileLines[j][3]),(0,60), font, 0.3,(255,255,255),1,cv2.LINE_AA)

    image_array = np.asarray([image])
    steering = model.predict(image_array, batch_size=1)
    cv2.putText(image,'Steering: '+ str(steering),(150,110), font, 3,(255,255,255),1,cv2.LINE_AA)
    
    cv2.imwrite('{}.jpg'.format(timestamp), (expanded_image+0.5)*int(255))



    print('Prediction: ' +str(steering) )


