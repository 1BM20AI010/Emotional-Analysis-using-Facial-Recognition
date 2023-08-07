Python 3.11.4 (tags/v3.11.4:d2340ef, Jun  7 2023, 05:45:37) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import os
... import numpy as np
... import pandas as pd
... import seaborn as sns
... from PIL import Image
... import tensorflow as tf
... from numpy import asarray
... import matplotlib.pyplot as plt
... from IPython.display import SVG, Image
... from tensorflow.keras.optimizers import Adam
... from tensorflow.keras.utils import plot_model
... from tensorflow.keras.models import Model, Sequential
... from tensorflow.keras.preprocessing.image import ImageDataGenerator
... from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
... from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D
... from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
... 
... img_size, batch_size = 48, 64
... 
... df = pd.read_csv("fer2013.csv")
... df.info()
... 
... df.head(2)
... 
... def emotions_mapping(row):
...     val = row["emotion"]
...     if val == 0:
...         return "Angry"
...     elif val == 1:
...         return "Disgust"
...     elif val == 2:
...         return "Fear"
...     elif val == 3:
...         return "Happy"
...     elif val == 4:
...         return "Sad"
...     elif val == 5:
        return "Surprise"
    else:
        return "Neutral"

df["actual_emotion"] = df.apply(emotions_mapping, axis=1)

df = df[df.emotion != 1]
ax = sns.catplot(x="actual_emotion", kind="count", palette="ch:.25", data=df, height=5, aspect=1.2)
ax.fig.suptitle("Emotions Distribution")
ax

def transform_pixels(row):

    val = list(row["pixels"].split(' '))
    val = np.asarray(val, dtype=np.uint8)
    val = val.reshape((img_size, img_size))
    return val
    df["numpy_pixels"] = df.apply(transform_pixels, axis=1)

train = df[df.Usage == "Training"]
train.info()

test  = df[df.Usage != "Training"]
test.info()

train_plot = train.groupby(["actual_emotion"]).head(6)
train_plot_0 = train_plot[train_plot.emotion == 0]
train_plot_2 = train_plot[train_plot.emotion == 2]
train_plot_3 = train_plot[train_plot.emotion == 3]
train_plot_4 = train_plot[train_plot.emotion == 4]
train_plot_5 = train_plot[train_plot.emotion == 5]
train_plot_6 = train_plot[train_plot.emotion == 6]

t_list = [train_plot_0, train_plot_2, train_plot_3, train_plot_4, train_plot_5, train_plot_6]

from PIL import Image

plt.figure(0, figsize=(14, 20))
ctr = 0

for tl in t_list:
    for i, row in tl.iterrows():
        ctr += 1
        plt.subplot(7, 6, ctr, title=row["actual_emotion"])
        val = list(row["pixels"].split(' '))
        val = np.asarray(val, dtype=np.uint8)
        val = val.reshape((img_size, img_size))
        img = Image.fromarray(val)
        plt.imshow(img, cmap='gray')

plt.tight_layout()
plt.show()

train.actual_emotion.value_counts()

test.actual_emotion.value_counts()

ax = sns.catplot(x="actual_emotion", kind="count", palette="ch:.25", data=train, height=5, aspect=1.2)
ax.fig.suptitle("Training Distribution")
ax

ax = sns.catplot(x="actual_emotion", kind="count", palette="ch:.34", data=test, height=5, aspect=1.2)
ax.fig.suptitle("Testing Distribution")
ax

model = Sequential()

# 1. Conv
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(img_size, img_size,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


# 2. Conv Layer
model.add(Conv2D(128, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 3. Conv Layer
model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#4. Conv Layer
model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(6, activation='softmax'))
model.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model1_json = model.to_json()

with open("model_v1.json", "w") as json_file1:
    json_file1.write(model1_json)

model2_json = model.to_json()

with open("model_v2.json", "w") as json_file1:
