Dear Alam,

We will try together to make a demo about Drift detection in videos. We need first to begin with the database collection. The code is already working, adaptable to any new dataset.
These are the steps for dataset building:
1- search on google  youtube videos tagged with ?????
2- download them using https://en.savefrom.net/1-youtube-video-downloader/
3- crop small sequence of code containing only the drift action (from 1 to 3 seconds) no more. We need 50 videos for drifting and 50 for no_drifting. You can use this python code:

########################################################################
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

video = VideoFileClip('video.avi').subclip(0, 10)
video.write_videofile('crop_video_1.mp4',audio_codec='aac')
########################################################################


4- convert every cropped video to separate frames following the naming convention. You can use this code:


################################################################################
#### script to split the video into multiple images
################################################################################

import cv2

vidcap = cv2.VideoCapture('/home/bilel/1-demos/AnomalyDetection/AVSS2019/Demo/test2.mp4')
result = vidcap.isOpened();
success,image = vidcap.read()
print("result = ", result);
count = 0
while success:
        print('1 Read a new frame: ', success)
        cv2.imwrite("output_folder/image_%05d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('2 Read a new frame: ', success)
        count += 1
################################################################################
5- Add for every video folder a file named n_frames containing the number of frames.
6- Check the global format of the dataset with respect to other datasets. If the dataset is ready, please update as a reply to this mail. 
7- Once we get the dataset ready, we can run the training and test the code.


Cordially,
Bilel