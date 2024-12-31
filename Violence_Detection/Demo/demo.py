import os
import sys
import cv2
from utils import sec_to_hms
from backend import Backend
from videoloader import VideoLoader, frames_to_tensor


def main():
    video_path = '/home/alam/Downloads/Project_for_Saudi_drifting/Violence_Detection/Demo/7.mp4' #sys.argv[1]

    if not os.path.exists(video_path):
        return

    be = Backend()
    vl = VideoLoader(video_path)

    fps = vl.fps
    ###########################################
    cap = cv2.VideoCapture(video_path)
    frame_counter =0
    frame_violence_probability = 0.13
    w = int(cap.get(3))
    h = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))
    #w=w-300
    #h= h+100
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue

            frame_counter +=1
            if(frame_counter%16==0):
                frames = vl.get_frames()
                x = frames_to_tensor(frames)
                y, prob_violence = be.predict(x)
                frame_violence_probability = 1- prob_violence
                

            
            colour = (0, int((1 -frame_violence_probability)*255), int(frame_violence_probability*255))
            cv2.putText(frame,'Drifting degree', (w-300, h-150), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
            percentage = str(int(frame_violence_probability *100)) + '%'
            cv2.putText(frame,percentage, (w-205, h-180), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
            cv2.rectangle(frame, (w-200, h-(220 + int(frame_violence_probability*200))), (w-150, h-220), colour, -1)
            cv2.rectangle(frame, (w-204, h-424), (w-146, h-216), colour, 2)
            cv2.imshow('video', frame)
            out.write(frame)
        if cv2.waitKey(10) == 27:
            break
    out.release()
    cv2.destroyAllWindows()
    ############################################
    # loop through the whole video
    # todo use threading to handle it when having gpus
    '''
    while True:
        frames = vl.get_frames()

        # vl.get_frames
        if frames is None:
            return

        x = frames_to_tensor(frames)
        y, prob_violence = be.predict(x)

        for label in y:
            if label == 'violent':
                time = vl.pos / fps
                h, m, s = sec_to_hms(time)
                print('violent scene (probability=%f) at time:\n%d:%d:%d' % (prob_violence, h, m, s))
        #cv2.imshow("Video", frames[10])
    '''

    


if __name__ == '__main__':
    main()