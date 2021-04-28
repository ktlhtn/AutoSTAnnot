# -*- coding: utf-8 -*-
from math import pi, acos, sqrt
import numpy as np
import csv



def projection_angle_azimuth(x_temp, d):
    """
    Project the x-coordinate of the image (relative to number of pixels) into an
    azimuth angle in radians.
    
    x_temp: The relative x-coordinate of the pixels, in the range [0,1]
    d: d=1.0 is a stereographic projection, d=0.0 is a perspective projection
    
    projected_angle: The azimuth angle in radians (i.e. in the range [-pi, pi],
                     or [-180,180] in degrees)
    """
    x = x_temp - 0.5 # We want the center pixel to have an angle of 0
    x_max = 0.5
    numerator = -2 * d * x ** 2 + 2 * (d + 1) * sqrt((1 - d ** 2) * x ** 2 + (d + 1) ** 2)
    denominator = 2 * (x ** 2 + (d + 1) ** 2)
    if 0 < x < x_max:
        projected_angle = acos(numerator / denominator) * 2*pi
    elif -x_max < x < 0:
        projected_angle = - acos(numerator / denominator) * 2*pi
    elif x == x_max:
        projected_angle = pi
    elif x == -x_max:
        projected_angle = -pi
    elif x == 0:
        projected_angle = 0.0
    else:
        print(x)
        print(x_temp)
        raise Exception('Invalid input arguments!')
        
    return projected_angle


def projection_angle_elevation(y_temp, d):
    """
    Project the y-coordinate of the image (relative to number of pixels) into an
    elevation angle in radians.
    
    y_temp: The relative y-coordinate of the pixels, in the range [0,1]
    d: d=1.0 is a stereographic projection, d=0.0 is a perspective projection
    
    projected_angle: The elevation angle in radians (i.e. in the range [-pi/2, pi/2],
                     or [-90,90] in degrees)
    """
    y = y_temp - 0.5 # We want the center pixel to have an angle of 0
    y_max = 0.5
    numerator = -2 * d * y ** 2 + 2 * (d + 1) * sqrt((1 - d ** 2) * y ** 2 + (d + 1) ** 2)
    denominator = 2 * (y ** 2 + (d + 1) ** 2)
    if 0 < y < y_max:
        projected_angle = - acos(numerator / denominator) * pi
    elif -y_max < y < 0:
        projected_angle = acos(numerator / denominator) * pi
    elif y == y_max:
        projected_angle = -pi/2
    elif y == -y_max:
        projected_angle = pi/2
    elif y == 0:
        projected_angle = 0.0
    else:
        print(y)
        raise Exception('Invalid input arguments!')
        
    return projected_angle



def bounding_box_centers_to_azimuth_elevation(input_csv_file):
    """
    Convert the bounding box center coordinates into azimuth and elevation.
    
    input_csv_file: The name (+ path if in another directory) of the input CSV file
    
    """
    
    header_row = None
    input_csv_data = []
    
    # Read the input CSV file. Perform some cleaning for the detections, since
    # some detection coordinates are larger than 1.0 in their relative coordinates,
    # meaning that the bounding boxes are not visible in reality.
    with open(input_csv_file, 'r') as csvfile:
        csvreader_original = csv.reader(csvfile, delimiter=',')
        n = 0
        for row in csvreader_original:
            if n == 0:
                header_row = row
                n += 1
            else:
                # Clean the CSV data
                if float(row[4]) > 1.0:
                    pass
                elif float(row[5]) > 1.0:
                    pass
                elif float(row[6]) > 1.0:
                    pass
                elif float(row[7]) > 1.0:
                    pass
                else:
                    input_csv_data.append(row)
    
    width_in_pixels = int(header_row[2])
    height_in_pixels = int(header_row[3])
    number_of_last_frame = int(input_csv_data[-1][0])
    
    # A dict where the frame indeces are the keys and the values of the keys
    # are two-value lists in the form [azimuth, elevation] for all of the
    # objects in each frame
    bbox_centers = {}
    
    # Go through all frames one by one
    frame_indeces = np.arange(1, number_of_last_frame+1)
    for frame_index in frame_indeces:
        index = str(frame_index)
        frame_elements = [] # All detections within one frame
        
        for item in input_csv_data:
            if item[0] == index:
                frame_elements.append(item)
        
        list_of_directions_in_frame = [] # A list which contains the two-value lists of azimuth and elevation
        
        for element in frame_elements:
            bb_center_x = float(element[4])
            bb_center_y = float(element[5])
            
            # Convert bounding box centers into azimuth and elevation
            d=1.0 # We use stereographic projection, d=0.0 would be a perspective projection
            azimuth = projection_angle_azimuth(bb_center_x, d)
            elevation = projection_angle_elevation(bb_center_y, d)
            
            list_of_directions_in_frame.append([azimuth,elevation])
        
        # Add all azimuth and elevation angles into a dict with the frame index
        # is the key
        bbox_centers[index] = list_of_directions_in_frame
    
    return bbox_centers
    


if __name__ == '__main__':
    
    # Test the function
    test = bounding_box_centers_to_azimuth_elevation('R01_10fps_csv_output_bbox_cleaned.csv')
