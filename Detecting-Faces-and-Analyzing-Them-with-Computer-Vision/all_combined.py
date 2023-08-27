import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np



model = tf.keras.models.load_model('face_emotion_model.h5')

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"


genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(20-28)', '(30-40)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

cascade_path = 'haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascade_path)
face_roi = None
video = cv2.VideoCapture(0)
Predictions = None
padding = 20
# ...
# ...
while True:
    ret, frame = video.read()
    faces = faceCascade.detectMultiScale(frame, 1.1, 4)
    
    # Check if any faces are detected
    if len(faces) == 0:
        continue
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi = frame[y:y + h, x:x + w]
        text_x = x + int(w/2)- 300
        text_y = y - 30



    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPred = genderNet.forward()
    gender = genderList[genderPred[0].argmax()]

    ageNet.setInput(blob)
    agePred = ageNet.forward()
    age = ageList[agePred[0].argmax()]
    label = "selam ben müdafer"

    final_image = cv2.resize(face_roi, (224, 224))

    # Check if final_image is empty
    if final_image.size == 0:
        continue

    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image / 255.0
    Predictions = model.predict(final_image)
    print(Predictions[0])




    print(gender)
    print(age)
    
    predicted_class = np.argmax(Predictions)
    font = cv2.FONT_HERSHEY_PLAIN
    if predicted_class == 0:
        status = "A {} y.old {} looks Angry".format(age,gender)
        x1,y1,w1,h1 = 0,0,175,175
        cv2.putText(frame,
                    status,
                    (text_x,text_y),
                    font,
                    3,
                    (0,0,255),
                    2,
                    cv2.LINE_4)
   
    elif predicted_class == 1:
        status = "A {} y.old {} looks Disguested".format(age,gender)
        x1,y1,w1,h1 = 0,0,175,175

        cv2.putText(frame,
                    status,
                    (text_x,text_y),
                    font,
                    3,
                    (0,0,255),
                    2,
                    cv2.LINE_4)
       
    elif predicted_class == 2:
        if(Predictions[0][2]<7):
            status = "A {} y.old {} looks Neutral".format(age,gender)
        else:    
            status = "A {} y.old {} looks Feared".format(age,gender)
        x1,y1,w1,h1 = 0,0,175,175


        cv2.putText(frame,
                    status,
                    (text_x,text_y),
                    font,
                    2,
                    (0,0,255),
                    2,
                    cv2.LINE_4)

    elif predicted_class == 3:
        status = "A {} y.old {} looks Happy".format(age,gender)
        x1,y1,w1,h1 = 0,0,175,175

        cv2.putText(frame,
                    status,
                    (text_x,text_y),
                    font,
                    2,
                    (0,0,255),
                    2,
                    cv2.LINE_4)

    elif predicted_class == 4:
        status = "A {} y.old {} looks Sad".format(age,gender)
        x1,y1,w1,h1 = 0,0,175,175

        cv2.putText(frame,
                    status,
                   (text_x,text_y),
                    font,
                    2,
                    (0,0,255),
                    2,
                    cv2.LINE_4)

    elif predicted_class == 5:
        status = "A {} y.old {} looks Suprised".format(age,gender)
        x1,y1,w1,h1 = 0,0,175,175

        cv2.putText(frame,
                    status,
                   (text_x,text_y),
                    font,
                    2,
                    (0,0,255),
                    2,
                    cv2.LINE_4)

    else:
        status = "A {} y.old {} looks Neutral".format(age,gender)
        x1,y1,w1,h1 = 0,0,175,175

        cv2.putText(frame,
                    status,
                   (text_x,text_y),
                    font,
                    2,
                    (0,0,255),
                    2,
                    cv2.LINE_4)
      
    cv2.imshow("webcam", frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()