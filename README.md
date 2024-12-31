# Drifting-Car-Classification
## Paper: Driftnet: Aggressive driving behaviour detection using 3d convolutional neural networks
Abstract:Aggressive driving (i.e., car drifting) is a dangerous behavior that puts human safety and life into a significant risk. In this paper, we targeted this issue and proposed DriftNet, a deep learning model to detect aggressive driving behavior automatically from videos. It is built upon a DenseNet 3D backbone. DrifNet has proven the ability to learn both spatial and temporal features related to car drifting by only training it on a weakly labeled dataset of traffic videos. We also created a dataset of car drifting in the Saudi Arabian context. The validation accuracy of DrifNet on this dataset was 77.5%, outperforming other tested algorithms. To the best of our knowledge, this is the first work that addresses the detection of aggressive driving behavior from traffic videos.
## Pre-trained Weights
[Weights for testing](link)
## Test with pre-trained weights
To test and validate the results please use the "weights in the above link" with the following code file "XXX". Moreover, the videos XXX.mp4 and XXX.mp4 can be used during testing. 
## Our results
Please check the video files "XXX.mp4" for drafting of cars using a Driftnet model performance. Moreover, here are the evaluations in term of accuracy and loss.
![image info](accuracy.png)
![image info](loss.png)
## How to Prepare the Custom dataset:
1. DataSet Path:
a) Raw Data folder: .../Dataset of Anomaly
b) Preprocessed Data Path: .../Violence_Detection/VioDB/drift_jpg.

2.  How to Prepare the dataset:
a) First Step is to crop the normal and drifting clips from video precisely by using:
.../crop.py or crop_1.py or crop_2.py
Note: You need to look for the start time of drifting/normal part of video then give path of that video in crop.py script and also time of starting in seconds. ( for example if drifting time is 1min +5sec then convert it in seconds =65seconds)
Note 2: It is better to crop every drifting and normal for 3 seconds and upon conversion to frames will become 48 to 50 frames.

b) Convert video to frame by using the script of:
.../convert_to_frames_copy.py

c) Remove the start frame from every converted videos frames. Because the code accept the frames from 00001 not 00000.jpg.

d) Create a n_frames.txt file in every folder of the video frames. Just write the number of frames in n_frames.txt file (For Example: If a folder 1_1 have 49 frames then just write 49 in n_frames.txt file and save it.

e) Repeat the above stages from (a) to (d) for every video and save draft video in one folder and no drifting video in another folder.

f) After preprocessed copy all framed folders of individual video to the VioDB folder with subfolder drifting_jpg. The drifting_jpg need to have two subfolder with name of “fi” and “no”. “fi” indicates the drifting and “no” is for no-drifitng.
 My computer path is:
 /home/alam/Downloads/Project_for_Saudi_drifting/Violence_Detection/VioDB/drift_jpg


## Training with Different Pre-trained models:
The training script is given in the path: .../Violence_Detection/VioNet/main.py or main_1.py (modified).

a) First Check from line:20-38 that which models are given for training: As these are the given models:
def main(config):
# load model
if config.model == 'c3d':
model, params = VioNet_C3D(config)
elif config.model == 'convlstm':
model, params = VioNet_ConvLSTM(config)
elif config.model == 'densenet':
model, params = VioNet_densenet(config)
elif config.model == 'densenet_lean':
model, params = VioNet_densenet_lean(config)
elif config.model == 'efficientnet_3d':
model, params = VioNet_efficientnet_3d(config)
elif config.model == 'VioNet_Res3D':
model, params = VioNet_efficientnet_3d(config)
elif config.model == 'VioNet_Res3D1':
model, params = VioNet_efficientnet_3d(config)
# default densenet
else:
model, params = VioNet_densenet_lean(config)

b) Change the Line= 64-65 for the model selection and activity name selection as :
'efficientnet_3d',#'efficientnet_3d', #efficientnet_3d, c3d, convlstm, densenet, densenet_lean
'drift',

c) Change the training parameters for given dataset in Line 175-200 as :
configs = {
'hockey': {
'lr': 1e-4,
'batch_size': 20
},
'movie': {
'lr': 1e-3,
'batch_size': 16
},
'vif': {
'lr': 1e-3,
'batch_size': 16
},
'saudi_fight':{
'lr': 1e-2,
'batch_size': 20
},
'mix':{
'lr': 1e-2,
'batch_size': 1
},
'drift':{
'lr':1e-2,
'batch_size': 1
}
}

d) Change the path for the dataset in Line-202 given and the dataset should be given in  VioDB folder with name (name_jpg folder).
for dataset in ['drift']: #['hockey', 'movie', 'vif', 'saudi_fight','mix']

e) In Last with specified environment required for this project and download pre-trianed weights from : - Download pretrained weights on Kinetices and put them into dir `weights`. [[Weights](https://drive.google.com/file/d/1pNrAzWHQJLzOEH_-407rel3VV45YuJ6f/view?usp=sharing)]
before running the main.py script file.


## Citation
@INPROCEEDINGS{9283799,
  author={Noor, Alam and Benjdira, Bilel and Ammar, Adel and Koubaa, Anis},
  booktitle={2020 First International Conference of Smart Systems and Emerging Technologies (SMARTTECH)}, 
  title={DriftNet: Aggressive Driving Behaviour Detection using 3D Convolutional Neural Networks}, 
  year={2020},
  volume={},
  number={},
  pages={214-219},
  keywords={Training;Three-dimensional displays;Roads;Safety;Automobiles;Security;Videos;Anomaly Detection;3D CNN;Car Drifting;Aggressive Driving;Abnormal driving;Violent driving},
  doi={10.1109/SMART-TECH49988.2020.00056}}

  ## Thanks
  The code is modified based on the [[github_repository](https://github.com/JimLee1996/AVSS2019/tree/master/src)]

