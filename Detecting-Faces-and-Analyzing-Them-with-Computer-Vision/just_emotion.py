import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('face_emotion_model.h5')

cascade_path = 'haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascade_path)
face_roi = None
video = cv2.VideoCapture(0)
Predictions = None
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi = frame[y:y + h, x:x + w]
        text_x = x + int(w/2) - 50
        text_y = y - 20
    final_image = cv2.resize(face_roi, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image / 255.0

    Predictions = model.predict(final_image)
    print(Predictions[0])

    font = cv2.FONT_HERSHEY_PLAIN
    if np.argmax(Predictions) == 0:
        status = "Angry"
        x1,y1,w1,h1 = 0,0,175,175
        cv2.rectangle(frame,(x1,x1), (x1+w1,y1+h1), (0,0,0),-1)
        cv2.putText(frame,
                    status,
                    (text_x,text_y),
                    font,
                    3,
                    (0,0,255),
                    2,
                    cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))       
    elif np.argmax(Predictions) == 1:
        status = "Disgust"
        x1,y1,w1,h1 = 0,0,175,175

        cv2.putText(frame,
                    status,
                    (text_x,text_y),
                    font,
                    3,
                    (0,0,255),
                    2,
                    cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))         
    elif np.argmax(Predictions) == 2:
        status = "Fear"
        x1,y1,w1,h1 = 0,0,175,175


        cv2.putText(frame,
                    status,
                    (text_x,text_y),
                    font,
                    3,
                    (0,0,255),
                    2,
                    cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255)) 
    elif np.argmax(Predictions) == 3:
        status = "Happy"
        x1,y1,w1,h1 = 0,0,175,175

        cv2.putText(frame,
                    status,
                    (text_x,text_y),
                    font,
                    3,
                    (0,0,255),
                    2,
                    cv2.LINE_4)

    elif np.argmax(Predictions) == 4:
        status = "Sad"
        x1,y1,w1,h1 = 0,0,175,175

        cv2.putText(frame,
                    status,
                   (text_x,text_y),
                    font,
                    3,
                    (0,0,255),
                    2,
                    cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255)) 
    elif np.argmax(Predictions) == 5:
        status = "Suprise"
        x1,y1,w1,h1 = 0,0,175,175

        cv2.putText(frame,
                    status,
                   (text_x,text_y),
                    font,
                    3,
                    (0,0,255),
                    2,
                    cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255)) 
    else:
        status = "Neutral"
        x1,y1,w1,h1 = 0,0,175,175

        cv2.putText(frame,
                    status,
                   (text_x,text_y),
                    font,
                    3,
                    (0,0,255),
                    2,
                    cv2.LINE_4)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))         
    cv2.imshow("webcam", frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()