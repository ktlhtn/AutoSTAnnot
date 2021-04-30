# -*- coding: utf-8 -*-
from math import pi, acos, sqrt
import numpy as np
import csv
import os



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
        raise Exception('Invalid input arguments!')
        
    return projected_angle



def beamformer_input_singleframe(input_csv_file, name_of_text_file, filesep=os.path.sep):
    """
    Convert the bounding box center coordinates into azimuth and elevation. Write the
    converted information into a text file for the beamformer in MATLAB. The beamformer
    handles the input video frames frame by frame.
    
    input_csv_file: The name (+ path if in another directory) of the input CSV file. The file does NOT have coordinates
                    transformed into azimuth and elevation.
    name_of_text_file : The name (+ path if in another directory) of the output text file
    filesep: The file separator string
    
    """
    
    # Initialize text file
    file = open(name_of_text_file, 'w')
    file.close()
    
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
    
    name_of_file = header_row[0].split(filesep)[-1]
    fps = header_row[1]
    width_in_pixels = header_row[2]
    height_in_pixels = header_row[3]
    number_of_last_frame = int(input_csv_data[-1][0])
    
    # Go through all frames one by one (frame indexing begins with 1 in MATLAB)
    frame_indeces = np.arange(1, number_of_last_frame+1)
    for frame_index in frame_indeces:
        index = str(frame_index)
        frame_elements = [] # All detections within one frame
        
        for item in input_csv_data:
            if item[0] == index:
                frame_elements.append(item)
        
        for element in frame_elements:
            bb_center_x = float(element[4])
            bb_center_y = float(element[5])
            
            # Convert bounding box centers into azimuth and elevation
            d=1.0 # We use stereographic projection, d=0.0 would be a perspective projection
            azimuth = projection_angle_azimuth(bb_center_x, d)
            elevation = projection_angle_elevation(bb_center_y, d)
            
            # Write into a text file in the format
            # FILENAME_FRAMEID_CLASSID_CLASSNAME_AZIMUTH_ELEVATION_FPS_WIDTH_HEIGHT
            # Note that frame indexing begins with 1 in MATLAB
            text = name_of_file + ';' + index + ';' + element[1] + ';' + element[2].split(':')[0] + ';' + \
                   str(azimuth) + ';' + str(elevation) + ';' + fps + ';' + width_in_pixels + ';' + height_in_pixels
                 
            with open(name_of_text_file, 'a') as f:
                f.write(text + '\n')

    



def beamformer_input_multiframe(input_csv_file, name_of_text_file, filesep=os.path.sep):
    """
    Convert the bounding box center coordinates into azimuth and elevation. Write the
    converted information into a text file for the beamformer in MATLAB. The output gives the time arcs of a given
    object in the scene. This is a prototype and works only if there is only one object in a scene.
    
    input_csv_file: The name (+ path if in another directory) of the input CSV file. The file does NOT have coordinates
                    transformed into azimuth and elevation.
    name_of_text_file : The name (+ path if in another directory) of the output text file
    filesep: The file separator string
    
    """
    
    # Initialize text file
    file = open(name_of_text_file, 'w')
    file.close()
    
    # Initialize a temporary text file (to be cleaned at the end)
    name_of_temporary_file = 'temporary.txt'
    file = open(name_of_temporary_file, 'w')
    file.close()
    
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
    
    name_of_file = header_row[0].split(filesep)[-1]
    fps = header_row[1]
    width_in_pixels = header_row[2]
    height_in_pixels = header_row[3]
    number_of_last_frame = int(input_csv_data[-1][0])
    
    
    # Initialize the bounding box center of previous frame
    previous_bb_center_x = None
    previous_bb_center_y = None
    
    # Initialize the frame index of the previous object instance
    previous_object_instance_frame = None
    
    # Initialize index of time arc
    time_arc_index = 1
    
    
    # Go through all frames one by one (frame indexing begins with 1 in MATLAB)
    frame_indeces = np.arange(1, number_of_last_frame+1)
    for frame_index in frame_indeces:
        index = str(frame_index)
        frame_elements = [] # All detections within one frame
        
        for item in input_csv_data:
            if item[0] == index:
                frame_elements.append(item)
        
        for element in frame_elements:
            bb_center_x = float(element[4])
            bb_center_y = float(element[5])
            
            # If the bounding box was present in the previous frame AND the bounding box
            # center has not moved significantly, we continue the time arc of that object
            if frame_index != 1:
                if previous_object_instance_frame == (frame_index-1):
                    if abs(bb_center_x - previous_bb_center_x) < 0.1:
                        if abs(bb_center_y - previous_bb_center_y) < 0.1:
                            previous_bb_center_x = bb_center_x
                            previous_bb_center_y = bb_center_y
                            previous_object_instance_frame = frame_index
                else:
                    previous_bb_center_x = bb_center_x
                    previous_bb_center_y = bb_center_y
                    previous_object_instance_frame = frame_index
                    time_arc_index += 1
            else:
                previous_bb_center_x = bb_center_x
                previous_bb_center_y = bb_center_y
                previous_object_instance_frame = frame_index
            
            
            # Convert bounding box centers into azimuth and elevation
            d=1.0 # We use stereographic projection, d=0.0 would be a perspective projection
            azimuth = projection_angle_azimuth(bb_center_x, d)
            elevation = projection_angle_elevation(bb_center_y, d)
            
            # Write into a text file in the format
            # FILENAME_FRAMEID_CLASSID_CLASSNAME_AZIMUTH_ELEVATION_FPS_WIDTH_HEIGHT_TIMEARCINDEX
            # Note that frame indexing begins with 1 in MATLAB
            text = name_of_file + ';' + index + ';' + element[1] + ';' + element[2].split(':')[0] + ';' + \
                   str(azimuth) + ';' + str(elevation) + ';' + fps + ';' + width_in_pixels + ';' + height_in_pixels + \
                   ';' + str(time_arc_index)
                 
            with open(name_of_temporary_file, 'a') as f:
                f.write(text + '\n')
    
    
    # Perform cleaning for short time arcs
    clean_short_time_arcs(name_of_text_file, name_of_temporary_file)



def clean_short_time_arcs(name_of_text_file, name_of_temporary_file, shortest_time_arc=5):
    """
    Clean time arcs from the CSV file that are shorter than five frames (default).
    
    name_of_text_file: The text file that is cleaned.
    name_of_temporary_file: The temporary text file where the uncleaned data is stored.
    
    """
    
    # Read the content of the text file into a list
    with open(name_of_temporary_file) as f:
        content = f.readlines()
        
    # Remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    
    # Loop through the content. Gather each time arc into a list. If the time
    # arc is shorter than shortest_time_arc, then we don't add it to name_of_text_file.
    time_arc_index = 1
    time_arc_list = []
    previous_time_arc_index = -1
    for item in content:
        row_content = item.split(';')
        current_time_arc_index = int(row_content[-1])
        
        # We are iterating for the last element of the list
        if item == content[-1]:
            if len(time_arc_list) >= shortest_time_arc:
                for row in time_arc_list:
                    element = row.split(';')
                    text = element[0] + ';' + element[1] + ';' + element[2] + ';' + element[3] + ';' + \
                        element[4] + ';' + element[5] + ';' + element[6] + ';' + element[7] + ';' + element[8] + \
                        ';' + str(time_arc_index)
                        
                    with open(name_of_text_file, 'a') as f:
                        f.write(text + '\n')
            
        
        elif previous_time_arc_index != current_time_arc_index:
            # We are on the first iteration of the file parsing
            if previous_time_arc_index == -1:
                time_arc_list.append(item)
                previous_time_arc_index = int(row_content[-1])
            
            # We have reached the end of a time arc, so we check that is the time arc
            # long enough for it to be stored
            elif len(time_arc_list) >= shortest_time_arc:
                for row in time_arc_list:
                    # Write into a text file in the format
                    # FILENAME_FRAMEID_CLASSID_CLASSNAME_AZIMUTH_ELEVATION_FPS_WIDTH_HEIGHT_TIMEARCINDEX
                    # Note that frame indexing begins with 1 in MATLAB
                    element = row.split(';')
                    text = element[0] + ';' + element[1] + ';' + element[2] + ';' + element[3] + ';' + \
                        element[4] + ';' + element[5] + ';' + element[6] + ';' + element[7] + ';' + element[8] + \
                        ';' + str(time_arc_index)
                        
                    with open(name_of_text_file, 'a') as f:
                        f.write(text + '\n')
                        
                time_arc_list = []
                time_arc_index += 1
                time_arc_list.append(item)
                previous_time_arc_index = int(row_content[-1])
                
            else:
                time_arc_list = []
                time_arc_list.append(item)
                previous_time_arc_index = int(row_content[-1])
            
        else:
            time_arc_list.append(item)
            previous_time_arc_index = int(row_content[-1])
        
    # Delete the temporary file
    if os.path.exists(name_of_temporary_file):
        os.remove(name_of_temporary_file)
            
    



#if __name__ == '__main__':
    
    # Test the function
    #beamformer_input_singleframe('R01_10fps_csv_output_bbox_cleaned.csv',
    #                                                      'test_singleframe.txt', '\\')
    
    #beamformer_input_multiframe('mehmet_walking_around_cleaned_mapped.csv',
    #                                                      'test_multiframe_cleaned.txt', '\\')
