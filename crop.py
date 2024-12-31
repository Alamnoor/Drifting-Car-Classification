
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

video = VideoFileClip('video.avi').subclip(0, 10)
video.write_videofile('crop_video_1.mp4',audio_codec='aac')

