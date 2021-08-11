# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:31:01 2021

@Nama Anggota Kelompok:
1. Radithya Airlangga (672019007)
2. Rafi Ardhan Fauzan (672019025)
"""
import numpy as np #NumPy memiliki kemampuan untuk membentuk objek N-dimensional array, yang mirip dengan list.
import cv2 #Memanggil modul opencv untuk proses input,simpan dan menampilkan citra/ image.
import os #Berinteraksi dengan sistem operasi

execution_path = os.getcwd() #fungsi pada import os python yang berfungsi untuk menampilkan letak direktori program python yang akan di simpan dalam sistem operasi tersebut.
while (True):
 os.system('cls')
 print("======PROGRAM DETEKSI GAMBAR=======")
 print("====================================")
 print("PETUNJUK :")
 print("1. Letakkan gambar yang mau diiuji dalam satu folder dengan file ini dan rename dengan nama pic.jpg")
 print("2. Silahkan menunggu hingga hasil sudah keluar.")
 pil = str(input("Press ' Y ' Continue OR 'N' To Exit...."))

 if pil == 'Y' or  pil == 'y' :
     # Load Yolo
     net = cv2.dnn.readNet("weight/yolov3.weights", "cfg/yolov3.cfg")
     classes = []
     with open("coco.names", "r") as f:
         classes = [line.strip() for line in f.readlines()]
     layer_names = net.getLayerNames()
     output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
     colors = np.random.uniform(0, 255, size=(len(classes), 3))

     # Load image
     img = cv2.imread("pic/pic4.jpg")
     img = cv2.resize(img, None, fx=0.5, fy=0.5)  # Image Resize
     height, width, channels = img.shape

     # Detecting objects
     blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

     net.setInput(blob)
     outs = net.forward(output_layers)

     # Showing informations on the screen
     class_ids = []
     confidences = []
     boxes = []
     for out in outs:
         for detection in out:
             scores = detection[5:]
             class_id = np.argmax(scores)
             confidence = scores[class_id]
             if confidence > 0.1:  # Accuracy
                 # Object detected
                 center_x = int(detection[0] * width)
                 center_y = int(detection[1] * height)
                 w = int(detection[2] * width)
                 h = int(detection[3] * height)

                 # Rectangle coordinates
                 x = int(center_x - w / 2)
                 y = int(center_y - h / 2)

                 boxes.append([x, y, w, h])
                 confidences.append(float(confidence))
                 class_ids.append(class_id)

     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
     print(indexes)
     font = cv2.FONT_HERSHEY_SIMPLEX

     for i in range(len(boxes)):
         if i in indexes:
             x, y, w, h = boxes[i]
             label = str(classes[class_ids[i]])
             color = colors[i]
             color = (255, 255, 255)
             rectangle_bgr = (255, 255, 255)  # background label
             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
             cv2.putText(img, label, (x, y + 20), font, 1, color, 2)

     cv2.imshow("Image", img)
     cv2.waitKey(0)
     cv2.destroyAllWindows()
 else:
    print("Exiting App....")
    break