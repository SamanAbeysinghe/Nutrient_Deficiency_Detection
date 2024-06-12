import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import random
import cv2
import os
import shutil
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization, RandomFlip, RandomRotation, RandomZoom, Rescaling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

dataset_dir = '/content/drive/MyDrive/Project_Images/50 Project/Train'

def load_random_imgs_from_folder(folder):
  plt.figure(figsize=(20,20))
  for i in range(5):
    file = random.choice(os.listdir(folder))
    image_path = os.path.join(folder, file)
    img=mpimg.imread(image_path)
    ax=plt.subplot(1,5,i+1)
    ax.title.set_text(file)
    plt.xlabel(f'Image Shape: {img.shape}')
    plt.imshow(img)

load_random_imgs_from_folder('/content/drive/MyDrive/Project_Images/50 Project/Train/Potassium')

labels=[]
for i in os.listdir(dataset_dir):
  labels+=[i]

print(labels)

IMG_SIZE = 180

def get_data(data_dir):
  data = []
  for label in labels:
    path = os.path.join(data_dir, label)
    class_num = labels.index(label)
    for img in os.listdir(path):
      try:
        if img[-3:] != 'txt':
          img_arr = cv2.imread(os.path.join(path, img))[...,::-1] 
          if img_arr.shape[:2] == (IMG_SIZE, IMG_SIZE):
            data.append([img_arr, class_num])
          else:
            resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
            data.append([resized_arr, class_num])
      except Exception as e:
        print(f"Error reading image {img}: {e}")
  for element in data:
    if element[0].shape[:2] != (IMG_SIZE, IMG_SIZE):
      raise ValueError(f"All images must have the same size. Found an image with shape {element[0].shape[:2]}")

  return data

dataset = get_data(dataset_dir)
np.random.shuffle(dataset)

len(dataset)

train = dataset[:738]
val = dataset[739:759]
test = dataset[760:]

train_df = pd.DataFrame(train,columns=['Feature','Label'])

l=[]
for i in train_df['Label']:
  l.append(labels[i])

plt.figure(figsize=(7,4))
sns.countplot(x=l);
plt.title('Plant Deficiency Classifiers');

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

np.array(x_train).max()

plt.figure(figsize = (5,5))
plt.imshow(train[-1][0])
plt.title(labels[train[-1][1]])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)

x_test = np.array(x_test) / 255.0
y_test = np.array(y_test)

x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape

type(y_train[0])

plt.figure(figsize=(20,20))

for i in range(8):
  img = x_train[i]
  ax=plt.subplot(1,8,i+1)
  ax.title.set_text(labels[y_train[i]])
  plt.xlabel(f'Image Shape: {img.shape}')
  plt.imshow(img)

def plot_before_after(img, filtered_img):
  plt.figure(figsize=(10,10))
  plt.subplot(121),plt.imshow(img),plt.title('Original')
  plt.xticks([]), plt.yticks([])
  plt.subplot(122),plt.imshow(filtered_img),plt.title('Filtered')
  plt.xticks([]), plt.yticks([])
  plt.show()

class Filters:
  def __init__(self, x_train):
    self.x_train = x_train

  def Gaussian_Blurr(self, kernel):
    self.kernel = kernel
    gauss_blurr = []
    for i in range(len(self.x_train)-1):
      f_img = cv2.GaussianBlur(self.x_train[i], self.kernel,0)
      gauss_blurr.append(f_img)
    return gauss_blurr

  def Median_Blurr(self, K):
    self.K = K
    median_blurr = []
    for i in range(len(self.x_train)-1):
      img = self.x_train[i].astype('float32') / 255.0
      f_img = cv2.medianBlur(img, self.K)
      median_blurr.append(f_img)
    return median_blurr

  def Bilateral_Blurr(self, diameter, sigmaColor, sigmaSpace):
    self.d = diameter
    self.sc = sigmaColor
    self.ss = sigmaSpace
    bilateral_blurr = []
    for i in range(len(self.x_train)-1):
      img = self.x_train[i].astype('float32') / 255.0
      f_img = cv2.bilateralFilter(img, self.d, self.sc, self.ss)
      bilateral_blurr.append(f_img)
    return bilateral_blurr

img_filter = Filters(x_train)

gauss_imgs = img_filter.Gaussian_Blurr((3,3))
plot_before_after(x_train[4], gauss_imgs[4])

medians = img_filter.Median_Blurr(3)
plot_before_after(x_train[2], medians[2])

bilateral_imgs = img_filter.Bilateral_Blurr(3, 11, 5)
plot_before_after(x_train[2], bilateral_imgs[2])

plt.imshow(x_train[0])

x_train[0].shape

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    featurewise_center=False,  
    samplewise_center=False,  
    featurewise_std_normalization=False,  
    samplewise_std_normalization=False,  
    zca_whitening=False, 
    rotation_range = 30,  
    zoom_range = 0, 
    width_shift_range=0,  
    height_shift_range=0, 
    horizontal_flip = True,  
    vertical_flip=False)  

train_iterator_1 = train_datagen.flow(x_train, y_train, batch_size=32)
val_iterator_1 = val_datagen.flow(x_val, y_val, batch_size=32)
print('Batches train=%d, test=%d' % (len(train_iterator_1), len(val_iterator_1)))

batchX, batchy = train_iterator_1.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

def learning_curve(model_fit, key='accuracy', ylim=(0.8, 1.01)):
    plt.figure(figsize=(12,6))
    plt.plot(model_fit.history[key])
    plt.plot(model_fit.history['val_' + key])
    plt.title('Learning Curve')
    plt.ylabel(key.title())
    plt.xlabel('Epoch')
    plt.ylim(ylim)
    plt.legend(['train', 'val'], loc='best')
    plt.show()

def confusion_matrix_plot(matrix, model):
    plt.figure(figsize=(12,10))
    cmap = "YlGnBu"
    ax= plt.subplot()
    sns.heatmap(matrix, annot=True, fmt='g', ax=ax, cmap=cmap); 

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(labels);
    ax.yaxis.set_ticklabels(labels);
    plt.show()

def cal_score(model, key):

    _, train_acc = model.evaluate(x_train/255.0, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    _, val_acc = model.evaluate(x_val/255.0, y_val, verbose=0)


    print('Train: %.3f, Test: %.3f, Val: %.3f' % (train_acc, test_acc, val_acc))
    yprobs = model.predict(x_test, verbose=0)
    yclasses = np.argmax(yprobs,axis=1)


    test_kappa = cohen_kappa_score(y_test, yclasses)
    print('Test Cohens kappa: %f' % test_kappa)
    print('\n')
    matrix = confusion_matrix(y_test, yclasses)
    print(matrix)
    print('\n')

    f1 = f1_score(y_test, yclasses, average='weighted')
    print(f'F1 Score: {f1}')
    print('\n')

    print(classification_report(y_test, yclasses, target_names=labels))

    if key==1:
        confusion_matrix_plot(matrix, model)

callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),
             ModelCheckpoint(filepath='weights/xcep_best1.h5', save_best_only=True)]

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.applications.xception import Xception

base_model = Xception(input_shape = (IMG_SIZE, IMG_SIZE, 3), 
                                include_top = False, 
                                weights = 'imagenet')

x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4, activation='softmax')(x)

xcep_model1 = Model(base_model.input, x)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.01)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

xcep_model1.compile(optimizer = optimizer,
                   loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                   metrics = ['accuracy'])

xcep_model1.summary()

xcep_hist = xcep_model1.fit_generator(train_iterator_1,epochs=20,verbose=1,validation_data=val_iterator_1, callbacks = callbacks)

learning_curve(xcep_hist,'loss', ylim=(0,2))
learning_curve(xcep_hist, 'accuracy', ylim=(0,1))

cal_score(xcep_model1, 1)

xcep_model1.save('/content/drive/MyDrive/Ac_model.h5')



