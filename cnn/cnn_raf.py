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


img_size = 100 #original size of the image
targetx = img_size
targety = img_size

epochs = 100
batch_size = 64

train_dir = "DATASET/train"
test_dir = "DATASET/test"

"""
Applying data augmentation to the images
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

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(targetx, targety),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=seed,

)
val_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(targetx, targety),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=seed,
     subset="training"

)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(targetx, targety),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    subset="validation",
        seed=seed
)

model= tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(targetx, targety, 3))) # đầu vào là ảnh RGB kích thước (100, 100, 3)
# Ảnh → Resize + Augmentation → CNN Blocks → Flatten → FC Layers → Softmax → Class

# Block 1: 2 lớp Conv2D với 32 và 64 filters. Sau đó chuẩn hóa và giảm chiều bằng MaxPooling.
# Conv2D(32) → Conv2D(64) → BatchNorm → MaxPooling → Dropout
# Học các đặc trưng cơ bản như cạnh, góc.
# Kích thước nhỏ, depth còn thấp → hoạt động như các bộ lọc đơn giản.
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 2: Sử dụng kernel lớn hơn (5x5) để tăng receptive field.
# Conv2D(128, kernel_size=5x5) → BatchNorm → MaxPooling → Dropout
# Kernel lớn hơn → receptive field rộng hơn → học được đặc trưng phức tạp hơn (vật thể, hoa văn...).
model.add(Conv2D(128,(5,5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 3 & 4: Dùng regularization L2 để chống overfitting. Tăng số filters lên 512 → trích xuất đặc trưng sâu hơn.
# Conv2D(512) + L2 regularization → BatchNorm → MaxPooling → Dropout
# Độ sâu lớn hơn → học đặc trưng trừu tượng, có tính khái quát cao.
# L2 giúp hạn chế mô hình "nhớ quá kỹ" (overfitting).
# BatchNorm ổn định và tăng tốc quá trình huấn luyện.
model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Mục tiêu các block CNN: từ đặc trưng cục bộ đơn giản → trích xuất đặc trưng toàn cục phức tạp hơn.

# Tầng phân loại Fully Connected (FC) layers:
# Flatten → Dense(256) → Dense(512) → Dense(7, softmax)
# Flatten: biến ảnh thành vector để đưa vào FC.
# Dense(256) → Dense(512): học quan hệ giữa đặc trưng và class, hoạt động như một “bộ suy luận”.
# softmax: xuất xác suất thuộc về mỗi lớp (7 lớp).
# Mục tiêu: đưa ra dự đoán xác suất cho mỗi lớp từ đặc trưng đã học.
model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(512,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

# Compile mô hình:
# Adam: tối ưu hóa hiệu quả.
# categorical_crossentropy: phù hợp với phân loại đa lớp (với nhãn one-hot).
# accuracy: để đo lường hiệu suất.

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
img = image.load_img("DATASET/test/4/test_0041_aligned.jpg",target_size = (targetx,targety,3))
img = np.array(img)
plt.imshow(img)
print(img.shape) #prints (48,48) that is the shape of our image

label_dict = {0:'SURPRISED',1:'FEARFUL',2:'DISGUSTED',3:'HAPPY',4:'SAD',5:'ANGRY',6:'NEUTRAL'}


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