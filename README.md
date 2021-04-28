AutoSTAnnot project (automated spatio-temporal training data annotation for spatial audio using 360 video detections)

An example on how to use the pipeline for creating annotated training data with these tools: 






Components:


----------------------------------------
Video detection from 360 video (mp-YOLO-video)

Modified from https://github.com/keevin60907/mp-YOLO, which is based on https://arxiv.org/abs/1805.08009

mp-YOLO was shared under the MIT-license (license-file under the directory mp-YOLO-video)

The code is based on the mp-YOLO project found from github. There is no license file added to the project, which means it can't be used publicly as such. The project source code was modified so that it takes in a video file as a parameter and applies the detections for each frame on the video. In addition, the code also outputs a .csv file that contains all the detections (detection id, label, confidence, bounding box center coordinates, bounding box width and height in relative coordinates) for each frame. The header (first line) of the produced .csv file has the fps and resolution of the processed video. 

For publicly using the projection part (from equirectangular 360 video to 4 projected sub images) without a lisence agreement from the original developer one should reimplement the projection, which is described in the research paper https://arxiv.org/abs/1805.08009

What it does: 

Reads in a 360 video file (equirectangular) and uses the yolov4 network to detects objects in the video frame by frame. Outputs the detections to a .csv file. 

How to use: 

python detect.py $ABSOLUTE_PATH_TO_VIDEO_FILE$

Outputs the processed video to directory 'processed' in the current working dir (cwd).
Outputs the detections file (.csv) to directory 'detections' in cwd.
----------------------------------------

----------------------------------------
Detections to powermap or video

What it does:

Reads in the detections .csv file and either draws the detected bounding boxes to the configured video file frames or crops a powermap .mat file generated with matlab from the audio data so that the powermap values within the bounding boxes are included and everything else is set to zero. 

How to use: 

python detections_to_powermap.py $ABSOLUTE_PATH_TO_POWERMAP_FILE$ (video or .mat) $ABSOLUTE_PATH_TO_DETECTIONS_FILE$ (.csv) $MODE$

Mode is either -v, -m or -mc. 

If -v is selected the program assumes the input file (powermap) is a video file and draws the bounding boxes to the video frames and outputs a processed video with bounding boxes visible. 

If -m is selected the program assumes the input file is a .mat powermap file and crops the powermap with the bounding boxes so that each detected class in the whole timespan of the data is detected to a class specific powermap. The output of this mode is therefore "processed_0_map.mat" and "processed_65_map.mat" for powermap frames that have person(s) (class id 0) and mouse(s) (class id 65) detected from the video recording.

If -mc is selected the program assumes the input file is a .mat powermap file and crops the powermap with the bounding boxes so that all the detected class instances are cropped in one single powermap. 
----------------------------------------

bbox_cleaner_and_map_classes.py

What it does: 

How to use: 

----------------------------------------


video_resampling

What it does:

Can be used to downsample the framerate of a video file. The video file from which the detections are done should have the shame framerate as the powermap produced from the audio data. 



----------------------------------------


FOA_Beamformer

What it does:

How to use:


----------------------------------------

FOA_powermap

what it does:

How to use: 

----------------------------------------