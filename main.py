import os
import cv2
import time
import threading
import sqlite3
import numpy as np
import face_recognition

path = 'image assets'
images = []
names = []
dir_list = os.listdir(path)

for cls in dir_list:
    cur_img = cv2.imread(f"{path}/{cls}")
    images.append(cur_img)
    names.append(os.path.splitext(cls)[0])

def find_encoding(images:list):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


def face_recog():
    encodelistknown = find_encoding(images)
    print("Encoding Complete")
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        face_frame = face_recognition.face_locations(imgs)
        encode_frame = face_recognition.face_encodings(imgs, face_frame)

        for encode_face, face_loc in zip(encode_frame, face_frame):
            matches = face_recognition.compare_faces(encodelistknown, encode_face)
            face_distance = face_recognition.face_distance(encodelistknown, encode_face)
            match_index = np.argmin(face_distance)
            if matches[match_index]:
                name = names[match_index].upper()
                print(name)
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x1, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("webcam", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    t1 = threading.Thread(target=face_recog(), args=())

'''sya = face_recognition.load_image_file('image assets\Shyam Saseendran.jpeg')
ada = face_recognition.load_image_file('image assets\Adarsh Sreenivasan.jpeg')
ada = cv2.cvtColor(ada, cv2.COLOR_BGR2RGB)
sya = cv2.cvtColor(sya, cv2.COLOR_BGR2RGB)

syloc = face_recognition.face_locations(sya)[0]
syenc = face_recognition.face_encodings(sya)[0]
cv2.rectangle(sya, (syloc[3], syloc[0]), (syloc[1], syloc[2]),(0, 255, 0), 2)


cv2.imshow("syam", sya)
cv2.imshow("adarsh", ada)
cv2.waitKey(0)  '''


