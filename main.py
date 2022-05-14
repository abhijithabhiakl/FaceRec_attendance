import cv2
import time
import sqlite3
import numpy as np
import face_recognition

sya = face_recognition.load_image_file('image assets\Shyam Saseendran.jpeg')
ada = face_recognition.load_image_file('image assets\Adarsh Sreenivasan.jpeg')
sya = cv2.cvtColor(sya, cv2.COLOR_BGR2RGB)
ada = cv2.cvtColor(ada, cv2.COLOR_BGR2RGB)

facloc = face_recognition.face_locations(sya)[0]
encodesya = face_recognition.face_encodings(sya)[0]
cv2.rectangle(sya, (facloc[3], facloc[0]), (facloc[1], facloc[2]), (0, 255, 0), 2)

facloc1 = face_recognition.face_locations(ada)[0]
encodeada = face_recognition.face_encodings(ada)[0]
cv2.rectangle(ada, (facloc1[3], facloc1[0]), (facloc1[1], facloc1[2]), (0, 255, 0), 2)

results = face_recognition.compare_faces(encodesya, [encodeada])
resdis = face_recognition.face_distance(encodesya, [encodeada])
print(results)
print(resdis)

cv2.imshow("syam", sya)
cv2.imshow("ada", ada)
cv2.waitKey(0)


