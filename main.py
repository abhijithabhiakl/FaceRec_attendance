import cv2
import time
import face-recognition
import sqlite3
import numpy as np
import face_recognition

sya = face_recognition.load_image_file('image assets\sya.jpeg')
ada = face_recognition.load_image_file('image assets\sada.jpeg')
sya = cv2.cvtColor(sya, cv2.COLOR_BGR2RGB)
ada = cv2.cvtColor(ada, cv2.COLOR_BGR2RGB)

cv2.imshow("syam", sya)
cv2.imshow("ada", ada)
cv2.waitKey(0)


