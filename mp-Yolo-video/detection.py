'''
Object Detection on Panorama pictures
Usage:
    $ pyhton3 detection.py <pano_picture> <output_picture>

    pano_picture(str)  : the pano pic file
    output_picture(str): the result picture
'''
import sys
import cv2
import numpy as np
from stereo import pano2stereo, realign_bbox
import os
import time
import matplotlib.pyplot as plt
import csv

CF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_RESOLUTION = (416, 416)

class Yolo():
    '''
    Packed yolo Netwrok from cv2
    '''
    def __init__(self):
        # get model configuration and weight
        model_configuration = 'yolov4.cfg'
        model_weight = 'yolov4.weights'

        # define classes
        self.classes = None
        class_file = 'coco.names'
        with open(class_file, 'rt') as file:
            self.classes = file.read().rstrip('\n').split('\n')

        net = cv2.dnn.readNetFromDarknet(
            model_configuration, model_weight)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(cv2.cuda.getCudaEnabledDeviceCount())
        self.yolo = net

        self.cf_th = CF_THRESHOLD
        self.nms_th = NMS_THRESHOLD
        self.resolution = INPUT_RESOLUTION
        self.outputwriter = None
        print('Model Initialization Done!')

    def set_outputwriter(self, outputwriter):
        self.outputwriter = outputwriter

    def detect(self, frame):
        '''
        The yolo function which is provided by opencv

        Args:
            frames(np.array): input picture for object detection

        Returns:
            ret(np.array): all possible boxes with dim = (N, classes+5)
        '''
        blob = cv2.dnn.blobFromImage(np.float32(frame), 1/255, self.resolution,
                                     [0, 0, 0], 1, crop=False)

        self.yolo.setInput(blob)
        layers_names = self.yolo.getLayerNames()
        output_layer =\
            [layers_names[i[0] - 1] for i in self.yolo.getUnconnectedOutLayers()]
        outputs = self.yolo.forward(output_layer)

        ret = np.zeros((1, len(self.classes)+5))
        for out in outputs:
            ret = np.concatenate((ret, out), axis=0)
        return ret

    def draw_bbox(self, frame, class_id, conf, left, top, right, bottom, center_x, center_y, height, width, frame_id):
        '''
        Drew a Bounding Box

        Args:
            frame(np.array): the base image for painting box on
            class_id(int)  : id of the object
            conf(float)    : confidential score for the object
            left(int)      : the left pixel for the box
            top(int)       : the top pixel for the box
            right(int)     : the right pixel for the box
            bottom(int)    : the bottom pixel for the box

        Return:
            frame(np.array): the image with bounding box on it
        '''
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        cv2.circle(frame, (center_x,center_y), radius=3, color=(0, 0, 255), thickness=3)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert(class_id < len(self.classes))
            label = '%s:%s' % (self.classes[class_id], label)
        print(conf)

        #Write the bounding box related detection to the csv file
        self.outputwriter.writerow([frame_id, class_id, label, conf, center_x, center_y, height, width])

        #Display the label at the top of the bounding box
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
        top = max(top, label_size[1])
        #cv2.rectangle(frame,
        #              (left, top - round(1.5*label_size[1])),
        #              (left + round(label_size[0]), top + base_line),
        #              (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame

    def nms_selection(self, frame, output):
        '''
        Packing the openCV Non-Maximum Suppression Selection Algorthim

        Args:
            frame(np.array) : the input image for getting the size
            output(np.array): scores from yolo, and transform into confidence and class

        Returns:
            class_ids (list)  : the list of class id for the output from yolo
            confidences (list): the list of confidence for the output from yolo
            boxes (list)      : the list of box coordinate for the output from yolo
            indices (list)    : the list of box after NMS selection

        '''
        print('NMS selecting...')
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]


        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class
        # with the highest score.
        class_ids = []
        confidences = []
        boxes = []
        for detection in output:
            #print(detection[0])
            #print(detection[1])
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CF_THRESHOLD:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height, center_x, center_y])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CF_THRESHOLD, NMS_THRESHOLD)
        
        return class_ids, confidences, boxes, indices

    def process_output(self, input_img, frames, frame_id):
        '''
        Main progress in the class.
        Detecting the pics >> Calculate Re-align BBox >> NMS selection >> Draw BBox

        Args:
            input_img(np.array): the original pano image
            frames(list)       : the results from pan2stereo, the list contain four spects of view

        Returns:
            base_frame(np.array): the input pano image with BBoxes
        '''
        height = frames[0].shape[0]
        width = frames[0].shape[1]
        first_flag = True
        outputs = None

        print('Yolo Detecting...')
        for face, frame in enumerate(frames):
            output = self.detect(frame)

            for i in range(output.shape[0]):
                output[i, 0], output[i, 1], output[i, 2], output[i, 3] =\
                realign_bbox(output[i, 0], output[i, 1], output[i, 2], output[i, 3], face)
            if not first_flag:
                outputs = np.concatenate([outputs, output], axis=0)
            else:
                outputs = output
                first_flag = False

        base_frame = input_img
        # need to inverse preoject
        class_ids, confidences, boxes, indices = self.nms_selection(base_frame, outputs)
        print('Painting Bounding Boxes..')
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            center_x = box[4]
            center_y = box[5]
            base_frame = self.draw_bbox(base_frame, class_ids[i], confidences[i],
                           left, top, left + width, top + height, center_x, center_y, height, width, frame_id)

        return base_frame

def main(inputfile, cwd):
    '''
    For testing now..
    '''

    my_net = Yolo()
    inputname = inputfile
    outputname = cwd+'\\processed\\'+'processed_'+inputname.split("\\")[-1]
    name = inputname.split("\\")[-1]
    name = name.split(".")[0]

    #initialize object for csv output
    output_file = open(cwd+'\\detections\\'+name+'_csv_output.csv', "w")

    #Capture video from inputfile 
    cap = cv2.VideoCapture(inputname)
    
    #Get video fps and resolution for output header
    fps_v = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #initialize csv writer object
    writer = csv.writer(output_file)
    
    #write header to outputfile
    writer.writerow([inputname, fps_v, width, height])

    #Set the writer object for detector object
    my_net.set_outputwriter(writer)

    processed_video = []

    
    starting_time = time.time()
    frame_id = 0
    try:
        while True:
            _, frame = cap.read()
            #frame = cv2.flip(frame, 1)
            
            frame_id += 1
            
            if frame is None:
                break 

            #if np.mod(frame_id, 1000) != 0:
            #    continue


            elapsed_time = time.time() - starting_time
            print("Elapsed time: "+str(elapsed_time))
            print("Frame: "+str(frame_id)+"/"+str(frames_total))
            projections = pano2stereo(frame)
            frame = my_net.process_output(frame, projections, frame_id)   

            
            #Frame by frame visualization if needed
            #fps = "{:.0f} FPS".format(frame_id/elapsed_time)
            #cv2.putText(frame, fps, (0, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,255,255))
            #cv2.putText(frame, 'Press q to quit', (1000, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,255))
            #cv2.imshow("Image", frame)

            processed_video.append(frame)

            #key = cv2.waitKey(1)
            #if key & 0xFF == ord('q'):
            #    break  # q to quit
    
    except KeyboardInterrupt:

        pass

        


    cap.release()
    cv2.destroyAllWindows()
    
    out = cv2.VideoWriter(outputname,cv2.VideoWriter_fourcc(*'mp4v'), fps_v, (width, height))
 
    for i in range(len(processed_video)):
        out.write(processed_video[i])
    out.release()
    output_file.close()
    #output_frame = my_net.process_output(input_pano, projections)
    #cv2.imwrite(sys.argv[2], output_frame)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("No input filepath given! (Give an absolute location of the you want to process)")
        exit()
    inputfile = sys.argv[1]

    cwd = str(os.getcwd())

    if not os.path.isdir(cwd+'\\processed'):
            os.mkdir(cwd+'\\processed')
    
    if not os.path.isdir(cwd+'\\detections'):
            os.mkdir(cwd+'\\detections')

    if os.path.isfile(inputfile):
        main(inputfile, cwd)

    else:
        print("Input filepath not found")
        exit()

