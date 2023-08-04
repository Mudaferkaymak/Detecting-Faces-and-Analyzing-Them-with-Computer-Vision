import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

image = cv2.imread('sadgirl.jpeg')

if image is None:
    print("An Error has occurred while opening the image")
else:
   
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
 
    predictions = DeepFace.analyze(image, enforce_detection=False)
    print(predictions)
    print("2")
if len(predictions) > 0:
    predictions = predictions[0]
   


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cascade_path = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)


faces = faceCascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)    


font = cv2.FONT_HERSHEY_SIMPLEX
reference_size = 500
ratio = image.shape[0]/reference_size

thickness = max(1,int(ratio))
font_size = max (0.5,ratio)


cv2.putText(image,
            predictions['dominant_emotion'] + " " + predictions['dominant_race'] + " " + predictions['dominant_gender'],
             (x-10, y - 10),
            font,font_size,
            (0,0,255),
            thickness,
            cv2.LINE_4);


plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

