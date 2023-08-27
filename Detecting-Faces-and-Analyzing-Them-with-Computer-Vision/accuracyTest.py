import os
import tensorflow as tf
import cv2
import numpy as np

DataDirectory = "test\\"

Classes = ["0", "1", "2", "3", "4", "5", "6"]

correctNumPerClass = {cls: 0 for cls in Classes}
totalNumPerClass = {cls: 0 for cls in Classes}

model = tf.keras.models.load_model('face_emotion_model.h5')

for category in Classes:
    path = os.path.join(DataDirectory, category)
    for img in os.listdir(path):
        image_arr = cv2.imread(os.path.join(path, img))
        image_arr = cv2.resize(image_arr, (224, 224))
        # Expand dimensions to create a batch of size 1
        image_arr = np.expand_dims(image_arr, axis=0)
        Predictions = model.predict(image_arr)
        predicted_class = np.argmax(Predictions)
        print(f"Prediction for {img}: {predicted_class} (True class: {category})")
        totalNumPerClass[category] += 1
        if int(predicted_class) == int(category):
            correctNumPerClass[category] += 1

accuracy_per_class = {cls: correctNumPerClass[cls] / totalNumPerClass[cls] for cls in Classes}
overall_accuracy = sum(correctNumPerClass.values()) / sum(totalNumPerClass.values())

print("Accuracy per class:")
for cls in Classes:
    print(f"Class {cls}: {accuracy_per_class[cls]:.2%}")

print("Overall accuracy: {:.2%}".format(overall_accuracy))
