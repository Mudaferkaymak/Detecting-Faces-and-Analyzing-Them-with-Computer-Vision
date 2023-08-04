import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt



def faceBox(faceNet,frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1 =int(detection[0,0,i,3] * frameWidth )
            y1 =int(detection[0,0,i,4] * frameHeight )
            x2 =int(detection[0,0,i,5] * frameWidth )
            y2 =int(detection[0,0,i,6] * frameHeight )
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
    return frame,bboxs



faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

img = cv2.imread('sadgirl.jpeg')

padding = 20



img,bboxs = faceBox(faceNet,img)
for bbox in bboxs:
    face = img[max(0,bbox[1]-padding):min(bbox[3]+padding,img.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, img.shape[1]-1)]
    blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPred = genderNet.forward()
    gender = genderList[genderPred[0].argmax()]
    
    result = DeepFace.analyze(img, actions = ['emotion'], enforce_detection = False)
    if len(result) > 0:
        result = result[0]

    ageNet.setInput(blob)
    agePred = ageNet.forward()
    age = ageList[agePred[0].argmax()]
    
    label = "A {}-year-old {} feels {}".format(age,gender,result['dominant_emotion'])

    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_x = max(bbox[0] + (bbox[2] - bbox[0]) // 2 - label_width // 2, 0)
    text_y = max(bbox[1] - 10 - label_height, 0)

    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()