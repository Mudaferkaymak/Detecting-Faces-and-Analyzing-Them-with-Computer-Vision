import cv2
from deepface import DeepFace

cascade_path = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

live = cv2.VideoCapture(0)


if not live.isOpened():
    raise IOError("An Error has occured, cannot open webcam!")

while True:
    ret, frame =  live.read()
    
    result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection = False)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.1,4)
    if len(result) > 0:
        result = result[0]
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Calculate the position for the text label
        text_x = x + int(w/2) - 50
        text_y = y - 20
        
        cv2.putText(frame,
                    result['dominant_emotion'],
                    (text_x, text_y),
                    font, 1.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_4)
        
  
    cv2.imshow("Webcam", frame)    
    if  cv2.waitKey(2) & 0xFF == ord('q'):
            break

live.release()
cv2.destroyAllWindows()
