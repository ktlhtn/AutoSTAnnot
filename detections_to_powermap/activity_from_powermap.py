import csv
import numpy as np

import bbox_centers_to_azimuth_elevation_v1 as bbxy2sph


class pmap_activity:
    '''
    A class for outputting audio activity thats calculated from bounding box cropped powermaps for detection instances
    '''
    def __init__(self, cwd, outputname, header, activity_threshold=1):


        self.output_file = open(cwd+'\\activity\\'+outputname+'_activity'+'_csv_output.csv', "w", newline='')

        self.writer = csv.writer(self.output_file)
        self.writer.writerow(header)

        self.activity_t = activity_threshold

    def close_writer_handle(self):

        self.output_file.close()


    def sum_bb_power(self, powermap_frame):

        pm_vector = np.concatenate(powermap_frame)

        pm_sum = np.sum(pm_vector)

        return pm_sum

    def write_out_activity(self, detection, frame):

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


