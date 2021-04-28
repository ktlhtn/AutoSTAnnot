import csv
import numpy as np

import bbox_centers_to_azimuth_elevation_v1 as bbxy2sph


class pmap_activity:
    '''
    A class for outputting audio activity thats calculated from bounding box cropped powermaps for detection instances
    '''
    def __init__(self, cwd, outputname, header, activity_threshold=1):

        '''
        Class constructor

        Inputs:
        cwd: String, the directory path of the current working directory
        outputname: String, the filename for the outputted csv file that has the detections listed in it
        header: String, The header line (the very first line) to be written to the detections .csv file, it has the filename, resolution and framerate
        of the original video from which the detections were made
        activity_threshold: A threshold parameter that is used to determine the audio activity state for the detection, if the sum of the cropped
        powermap is over this threshold, the activity is set to one. Zero otherwise. 

        '''

        #Define output file, open file handle
        self.output_file = open(cwd+'\\activity\\'+outputname+'_activity'+'_csv_output.csv', "w", newline='')

        #Initialize csv writer
        self.writer = csv.writer(self.output_file)
        self.writer.writerow(header)

        self.activity_t = activity_threshold

    def close_writer_handle(self):
        #close file handle
        self.output_file.close()


    def sum_bb_power(self, powermap_frame):

        '''
        Sum the energy of a powermap array (all the values in the array summed into one)
        inputs:
        powermap_frame, np.ndarray
        
        '''

        pm_vector = np.concatenate(powermap_frame)

        pm_sum = np.sum(pm_vector)

        return pm_sum

    def write_out_activity(self, detection, frame):

        '''
        Calculates the energy sum for frame and determines the audio activity state for the detection by comparing the summed value to the threshold
        value.

        inputs:
        detection, a single yolo detection array for the frame (frame_id)

        Adds the spherical coordinates of the bounding box center coordinates and the audio activity state to the detection information before
        outputting the detection as a new line to the output csv-file
        '''

        frame_id = detection[0]
        class_id = detection[1]
        label = detection[2]
        conf = detection[3]
        center_x = float(detection[4])
        center_y = float(detection[5])
        height = float(detection[6])        
        width = float(detection[7])

        d = 1.0

        center_azim = bbxy2sph.projection_angle_azimuth(center_x, d)
        center_elev = bbxy2sph.projection_angle_elevation(center_y, d)

        pm_sum = self.sum_bb_power(frame)

        if pm_sum >= self.activity_t:
            activity = 1

        elif pm_sum < self.activity_t:
            activity = 0

        print("Cropped powermap sum: ")
        print(pm_sum)

        self.writer.writerow([frame_id, class_id, label, conf, center_x, center_y, height, width, center_azim, center_elev, activity])


