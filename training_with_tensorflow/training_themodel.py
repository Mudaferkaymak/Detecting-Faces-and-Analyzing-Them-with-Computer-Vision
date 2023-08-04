import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


Datadirectory = "train\\"
Classes = ["0", "1", "2", "3", "4", "5", "6"]

img_size = 224

training_data = []
counter = 0
def createtrainingset():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))
                new_arr = cv2.resize(img_arr, (img_size, img_size))
                training_data.append([new_arr, class_num])
            except Exception as e:
                pass

createtrainingset()

print(len(training_data))

random.shuffle(training_data)

X = []  # Images (features)
y = []  # Labels

for feature, label in training_data:
    X.append(feature)
    y.append(label)

y = np.array(y)
X = np.array(X)
X = X.reshape(-1, img_size, img_size, 3)
X = X / 255.0  # Normalize the image data between 0 and 1

print(X.shape)
print(y.shape)

plt.imshow(X[0])
plt.show()



model = tf.keras.applications.MobileNetV2()

#TRANSFER LEARNING  - TUNING ,weights will start from lasr check point 

base_input = model.layers[0].input
base_output = model.layers[-2].output

final_output = layers.Dense(128)(base_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation = 'softmax')(final_output)

new_model = keras.Model(inputs = base_input, outputs = final_output)

new_model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

new_model.fit(X,y, epochs=10, batch_size = 8)

new_model.save('onbes_epoch.h5')

