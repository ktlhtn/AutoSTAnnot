AutoSTAnnot project (automated spatio-temporal training data annotation for spatial audio using 360 video detections)

Components:


####
Video detection from 360 video (mp-YOLO-video)

Modified from https://github.com/keevin60907/mp-YOLO, which is based on https://arxiv.org/abs/1805.08009

What it does: 

Reads in a 360 video file (equirectangular) and uses the yolov4 network to detects objects in the video frame by frame. Outputs the detections to a .csv file. 

How to use: 

python detect.py $ABSOLUTE_PATH_TO_VIDEO_FILE$

Outputs the processed video to directory 'processed' in the current working dir (cwd).
Outputs the detections file (.csv) to directory 'detections' in cwd.
###

###
Detections to powermap or video

What it does:

Reads in the detections .csv file and either draws the detected bounding boxes to the configured video file frames or crops a powermap .mat file generated with matlab from the audio data so that the powermap values within the bounding boxes are included and everything else is set to zero. 

How to use: 

python detections_to_powermap.py $ABSOLUTE_PATH_TO_POWERMAP_FILE$ (video or .mat) $ABSOLUTE_PATH_TO_DETECTIONS_FILE$ (.csv) $MODE$

Mode is either -v or -m. If -v is selected the program assumes the input file (powermap) is a video file and draws the bounding boxes to the video frames. If -m is selected the program assumes the input file is a .mat powermap file and crops the powermap with the bounding boxes. (not yet done)
