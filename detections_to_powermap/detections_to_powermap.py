import sys
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import csv
import scipy.io


def draw_bbox(frame, detections, frame_v_id):

    frame_shape = np.shape(frame)
    frame_width = frame_shape[1]
    frame_height = frame_shape[0]

    for detection in detections:

        frame_d_id = detection[0]
        class_id = detection[1]
        label = detection[2]
        conf = detection[3]
        center_x = float(detection[4])
        center_y = float(detection[5])
        height = float(detection[6])        
        width = float(detection[7])

        center_x = int(center_x*frame_width)
        center_y = int(center_y*frame_height)
        height = int(height*frame_height)
        width = int(width*frame_width)

        if class_id == str(0):

            top_left_corner = (int(center_x-0.5*width), int(center_y-0.5*height))
            bottom_right_corner = (int(center_x+0.5*width), int(center_y+0.5*height))

            text_location = (int(center_x-0.5*width), int(center_y-0.5*height)+5)
            

            #print(top_left_corner)
            #print(bottom_right_corner)

            frame = cv2.rectangle(frame, top_left_corner, bottom_right_corner, (0,0,0), 3)
      
            frame = cv2.putText(frame, label, text_location, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)

    return frame



def crop_powermap(frame, detections):
    
    frame_shape = np.shape(frame)
    frame_width = frame_shape[1]
    frame_height = frame_shape[0]

    for detection in detections:

        frame_d_id = detection[0]
        class_id = detection[1]
        label = detection[2]
        conf = detection[3]
        center_x = float(detection[4])
        center_y = float(detection[5])
        height = float(detection[6])        
        width = float(detection[7])

        center_x = int(center_x*frame_width)
        center_y = int(center_y*frame_height)
        height = int(height*frame_height)
        width = int(width*frame_width)


        if class_id == str(0):

            print("Human detected!")

            top_left_corner = (int(center_x-0.5*width), int(center_y-0.5*height))
            bottom_right_corner = (int(center_x+0.5*width), int(center_y+0.5*height))

            x1 = top_left_corner[0]
            y1 = top_left_corner[1]

            x2 = bottom_right_corner[0]
            y2 = bottom_right_corner[1]

            frame[0:x1-1, :] = 0
            frame[x2+1:-1,:] = 0

            frame[:, y2+1:-1] = 0
            frame[:, 0:y1-1] = 0

    return frame





def main(inputfile, detections, cwd, mode):

    inputname = inputfile
    outputname = cwd+'\\processed\\'+'processed_'+inputname.split("\\")[-1]
    name = inputname.split("\\")[-1]
    name = name.split(".")[0]
    #initialize object for csv reading
    detections = open(detections, "r")
    detections_list = []
    detections_frame = []
    #initialize reader object
    reader = csv.reader(detections)
    header = next(reader)
    #Get fps and resolution of the video from which the detections have been made
    fps_d = header[1]
    width_d = header[2]
    height_d = header[3]
    starting_time = time.time()
    frame_id = 0


    #Read in videofile, draw bounding boxes from csv file detections per frame and output the video to a new file
    if mode == '-v':
        
        #Capture video from inputfile 
        cap = cv2.VideoCapture(inputname)
        
        #Get video fps and resolution for output header
        fps_v = int(cap.get(cv2.CAP_PROP_FPS))
        width_v  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        height_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_total_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        processed_video = []
        
        #Read in ALL the detections from the csv file
        for line in reader:
            if len(line) > 0:
                detections_list.append(line)
        
        while True:
            _, frame = cap.read()
            #frame = cv2.flip(frame, 1)
            detections_frame = []
            
            frame_id += 1

            #Read in the detections in the csv file that are from this frame
            for det in detections_list:
                if int(det[0]) == frame_id:
                    detections_frame.append(det)
            
            if frame is None:
                break 

            elapsed_time = time.time() - starting_time
            print("Elapsed time: "+str(elapsed_time))
            print("Frame: "+str(frame_id)+"/"+str(frames_total_v))
            frame = draw_bbox(frame, detections_frame, frame_id)  
            processed_video.append(frame)

        cap.release()
        cv2.destroyAllWindows()
        
        out = cv2.VideoWriter(outputname,cv2.VideoWriter_fourcc(*'mp4v'), fps_v, (width_v, height_v))
     
        for i in range(len(processed_video)):
            out.write(processed_video[i])
            print("Outputting video, frame: "+str(i)+"/"+str(len(processed_video)))
        out.release()
    
    #Read in a matlab .mat file and crop the content using the detections / bounding boxes made from each
    if mode == '-m':
        mat = scipy.io.loadmat(inputfile, matlab_compatible=True)

        powermap = mat["map"]
        powermap_scaled = mat["map_scaled"]

        frames_length = np.shape(powermap)[2]

        #Read in all the detections from the csv file
        for line in reader:
            if len(line) > 0:
                detections_list.append(line)

        for i in range(frames_length):
            frame_map = powermap[:,:,i]
            frame_map_scaled = powermap_scaled[:,:,i]
            detections_frame = []

            frame_id += 1

            elapsed_time = time.time() - starting_time
            print("Elapsed time: "+str(elapsed_time))
            print("Frame: "+str(frame_id)+"/"+str(frames_length))

            #The detections in the csv file that are from this frame
            for det in detections_list:
                if int(det[0]) == frame_id:
                    detections_frame.append(det)

            powermap[:,:,i] = crop_powermap(frame_map, detections_frame)

            powermap_scaled[:,:,i] = crop_powermap(frame_map_scaled, detections_frame)
           
        if np.shape(powermap) != np.shape(mat["map"]):
            print("Cropped powermap dimensions do not match")
            exit()

        if np.shape(powermap_scaled) != np.shape(mat["map_scaled"]):
            print("Cropped powermap (scaled) dimensions do not match")
            exit()

        mat["map"] = powermap
        mat["map_scaled"] = powermap_scaled

        scipy.io.savemat(outputname, mat)


    
    detections.close()       

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("No powermap filepath given! (Give an absolute location of the you want to process)")
        print("How to use: python detections_to_powermap.py path_to_powermap path_to_detections mode")
        exit()

    if len(sys.argv) < 3:
        print("No detections filepath given! (Give an absolute location of the you want to process)")
        print("How to use: python detections_to_powermap.py path_to_powermap path_to_detections mode")
        exit()

    if len(sys.argv) < 4:
        print("No mode filepath given! (mode -m (.mat powermap-file) or -v (video-file))")
        print("How to use: python detections_to_powermap.py path_to_powermap path_to_detections mode")
        exit()

    if len(sys.argv) > 4:
        print("Too many arguments")
        print("How to use: python detections_to_powermap.py path_to_powermap path_to_detections mode")
        exit()

    inputfile = sys.argv[1]
    detections = sys.argv[2]
    mode = sys.argv[3]

    cwd = str(os.getcwd())

    if not os.path.isdir(cwd+'\\processed'):
            os.mkdir(cwd+'\\processed')

    if os.path.isfile(inputfile):
        main(inputfile, detections, cwd, mode)

    else:
        print("Input filepath not found")
        exit()

