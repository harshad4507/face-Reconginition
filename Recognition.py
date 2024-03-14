import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = 'images'
images = []
personName = []
myList = os.listdir(path)
print(myList)

for cu_img in myList:
    current_image = cv2.imread(f'{path}/{cu_img}')
    images.append(current_image)

    personName.append(os.path.splitext(cu_img)[0])
print(personName)

#face encoding
def faceEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistKnow = faceEncodings(images)
print("All encoding Completes.")

def attendance(name):
    with open('recoginition.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            time_now = datetime.now()
            tstr = time_now.strftime('%H:%M:%S')
            dstr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{tstr},{dstr},\n')

#Finding the faces using the camera reding
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    faces = cv2.resize(frame,(0,0),None,0.25,0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame =face_recognition.face_encodings(faces,facesCurrentFrame)

    for encodeFace,facelocation in zip(encodesCurrentFrame,facesCurrentFrame):
        matches = face_recognition.compare_faces(encodelistKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodelistKnow,encodeFace)
        matchIndex = np.argmin(faceDis)
        # print(matches)
        # print(faceDis)pb
        # print(matches[matchIndex])

        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1 = facelocation
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            attendance(name)

    cv2.imshow("camera",frame)
    if cv2.waitKey(10) == 13:
        break

cap.release()
cv2.destroyAllWindows()
