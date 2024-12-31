import cv2
import os 

vidcap = cv2.VideoCapture('/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/46_1.mp4')
result = vidcap.isOpened()
success,image = vidcap.read()
print("result = ", result)
count = 0
os.mkdir("/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/46_1")
while success:
        print('1 Read a new frame: ', success)

        cv2.imwrite("/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/46_1/image_%05d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('2 Read a new frame: ', success)
        count += 1
#########
vidcap = cv2.VideoCapture('/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/47_1.mp4')
result = vidcap.isOpened()
success,image = vidcap.read()
print("result = ", result)
count = 0
os.mkdir("/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/47_1")
while success:
        print('1 Read a new frame: ', success)

        cv2.imwrite("/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/47_1/image_%05d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('2 Read a new frame: ', success)
        count += 1

####
vidcap = cv2.VideoCapture('/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/48_1.mp4')
result = vidcap.isOpened()
success,image = vidcap.read()
print("result = ", result)
count = 0
os.mkdir("/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/48_1")
while success:
        print('1 Read a new frame: ', success)

        cv2.imwrite("/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/48_1/image_%05d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('2 Read a new frame: ', success)
        count += 1
#####
vidcap = cv2.VideoCapture('/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/49_1.mp4')
result = vidcap.isOpened()
success,image = vidcap.read()
print("result = ", result)
count = 0
os.mkdir("/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/49_1")
while success:
        print('1 Read a new frame: ', success)

        cv2.imwrite("/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/49_1/image_%05d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('2 Read a new frame: ', success)
        count += 1



#####
vidcap = cv2.VideoCapture('/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/50_1.mp4')
result = vidcap.isOpened()
success,image = vidcap.read()
print("result = ", result)
count = 0
os.mkdir("/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/50_1")
while success:
        print('1 Read a new frame: ', success)

        cv2.imwrite("/home/alam/Downloads/Project_for_Saudi_drifting/Dataset of Anomaly/No_drifting/50_1/image_%05d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('2 Read a new frame: ', success)
        count += 1        