import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
import cv2
from tensorflow.keras.applications import VGG16, InceptionResNetV2
from keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax

# from mtcnn import MTCNN
from sklearn.metrics import classification_report, confusion_matrix
import itertools

import random
seed = random.randint(1, 1000)
print(seed)

# CHIA DỮ LIỆU THÀNH train / val / test (70/15/15)
import os, shutil, random
from tqdm import tqdm

SOURCE_DIR = "CK+48"
TARGET_DIR = "CK_split"
SPLIT_RATIOS = (0.7, 0.15, 0.15)

for split in ['train', 'val', 'test']:
    for label in os.listdir(SOURCE_DIR):
        os.makedirs(os.path.join(TARGET_DIR, split, label), exist_ok=True)

for label in tqdm(os.listdir(SOURCE_DIR)):
    label_dir = os.path.join(SOURCE_DIR, label)
    if not os.path.isdir(label_dir): continue

    files = os.listdir(label_dir)
    random.shuffle(files)

    total = len(files)
    train_end = int(SPLIT_RATIOS[0]*total)
    val_end = train_end + int(SPLIT_RATIOS[1]*total)

    splits = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

    for split, split_files in splits.items():
        for f in split_files:
            src = os.path.join(label_dir, f)
            dst = os.path.join(TARGET_DIR, split, label, f)
            shutil.copy2(src, dst)

print("Dữ liệu đã được chia.")

img_size = 48 #original size of the image
targetx = img_size
targety = img_size

epochs = 100
batch_size = 64

train_dir = "CK_split/train"
val_dir = "CK_split/val"
test_dir = "CK_split/test"

"""
Data Augmentation
--------------------------
"""

train_datagen = ImageDataGenerator(
        rescale=1./255,
        brightness_range=[0.9,1.1],
        horizontal_flip=True,
        fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.5
)

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',      # Theo dõi loss của tập validation
    patience=100,              # Dừng nếu val_loss không giảm trong 12 epoch liên tiếp
    restore_best_weights=True  # Trở về trọng số tốt nhất trước khi overfitting
)


train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(targetx, targety),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=seed,

)
val_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(targetx, targety),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=seed,
    #  subset="training"

)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(targetx, targety),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    # subset="validation",
        seed=seed
)

model= tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(targetx, targety, 3)))
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(5,5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(512,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

model.compile(
    optimizer = Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )
model.summary()

fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
fig.set_size_inches(12,4)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Training Loss vs Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

plt.show()

# MA TRẬN NHẦM LẪN VÀ CÁC CHỈ SỐ ĐÁNH GIÁ
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

import keras.utils as image
img = image.load_img("CK+48/happy/S014_005_00000015.png",target_size = (targetx,targety,3))
img = np.array(img)
plt.imshow(img)
print(img.shape)

label_dict = {0:'ANGER',1:'CONTEMPT',2:'DISGUST',3:'FEAR',4:'HAPPY',5:'SADNESS',6:'SURPRISE'}

img = np.expand_dims(img,axis = 0) #makes image shape (1,48,48)
img = img.reshape(1,targetx,targety,3)
result = model.predict(img)
result = list(result[0])
print(result)

img_index = result.index(max(result))
print(label_dict[img_index])
plt.show()

train_loss, train_acc = model.evaluate(train_generator)
val_loss, val_acc   = model.evaluate(val_generator)
print("final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(train_acc*100, val_acc*100))

test_loss, test_acc   = model.evaluate(test_generator)
print("final test accuracy = {:.2f}".format(test_acc*100))