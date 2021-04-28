# -*- coding: utf-8 -*-
import csv
import numpy as np
import sys
from shapely.geometry import Polygon


def calculate_iou(box_1, box_2):
    """
    Calculate the overlap of two bounding boxes. Please note that this is the
    intersection over union, so even if one bounding box is inside another,
    the overlap is not 1.0.
    
    """

    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    
    # Make sure that we are not dividing by zero
    if poly_1.union(poly_2).area > 0:
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    else:
        iou = 0.0
    
    return iou




def bbox_cleaner(input_csv_file, output_csv_file, overlap_threshold=0.000001):
    """
    Clean overlapping bounding boxes from the video object detector if a given
    overlap threshold is exceeded.
    
    input_csv_file: The name (+ path if in another directory) of the input CSV file
    output_csv_file: The name (+ path if in another directory) of the output CSV file
    overlap_threshold: If two overlapping bounding boxes for the same class ID overlap
                       more than overlap_threshold, then the bounding box with a lower
                       confidence value will discarded. Value is between [0,1]. The
                       greater the threshold, the less strict is the choice of dropping
                       away bounding boxes. E.g. a threshold of 0.0 will leave only one
                       instance of a given class present in a given frame. A threshold of
                       0.000001 will not drop out bounding boxes of the same class that
                       are present in the same frame if there is NO overlap at all between
                       the bounding boxes of these classes.
    
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
    
    #overlaps = [] # Uncomment for testing
    
    
    # Go through all frames one by one
    cleaned_csv_data = []
    frame_indeces = np.arange(1, number_of_last_frame+1)
    for frame_index in frame_indeces:
        index = str(frame_index)
        frame_elements = [] # All detections within one frame
        
        for item in input_csv_data:
            if item[0] == index:
                frame_elements.append(item)
                
        if len(frame_elements) > 0:
            non_unique_classes_in_frame = []
            for item in frame_elements:
                non_unique_classes_in_frame.append(item[1])
            unique_classes_in_frame = np.unique(non_unique_classes_in_frame)
            
            # For each unique class in the frame, perform cleaning
            for class_id in unique_classes_in_frame:
                
                # Gather all the elements in the frame with the same id
                same_classes_in_frame = []
                for element in frame_elements:
                    if element[1] == class_id:
                        same_classes_in_frame.append(element)
                        
                if len(same_classes_in_frame) > 1:
                    while len(same_classes_in_frame) > 1:
                        compared_element_1 = same_classes_in_frame[0]
                        compared_element_2 = same_classes_in_frame[1]
                        confidence_compared_element_1 = float(compared_element_1[3])
                        confidence_compared_element_2 = float(compared_element_2[3])
                        
                        
                        # The bounding box measures are converted to pixels
                        bb_center_x_compared_element_1 = float(compared_element_1[4]) * width_in_pixels
                        bb_center_y_compared_element_1 = float(compared_element_1[5]) * height_in_pixels
                        bb_height_compared_element_1 = float(compared_element_1[6]) * height_in_pixels
                        bb_width_compared_element_1 = float(compared_element_1[7]) * width_in_pixels
                        
                        bb_center_x_compared_element_2 = float(compared_element_2[4]) * width_in_pixels
                        bb_center_y_compared_element_2 = float(compared_element_2[5]) * height_in_pixels
                        bb_height_compared_element_2 = float(compared_element_2[6]) * height_in_pixels
                        bb_width_compared_element_2 = float(compared_element_2[7]) * width_in_pixels
                        
                        # The bounding boxes in the form [top_left, top_right, bottom_right, bottom_left]
                        bbox_element_1 = [[int(bb_center_x_compared_element_1 - bb_width_compared_element_1/2),
                                           int(bb_center_y_compared_element_1 - bb_height_compared_element_1/2)],
                                          [int(bb_center_x_compared_element_1 + bb_width_compared_element_1/2),
                                           int(bb_center_y_compared_element_1 - bb_height_compared_element_1/2)],
                                          [int(bb_center_x_compared_element_1 + bb_width_compared_element_1/2),
                                           int(bb_center_y_compared_element_1 + bb_height_compared_element_1/2)],
                                          [int(bb_center_x_compared_element_1 - bb_width_compared_element_1/2),
                                           int(bb_center_y_compared_element_1 + bb_height_compared_element_1/2)]]
                        
                        bbox_element_2 = [[int(bb_center_x_compared_element_2 - bb_width_compared_element_2/2),
                                           int(bb_center_y_compared_element_2 - bb_height_compared_element_2/2)],
                                          [int(bb_center_x_compared_element_2 + bb_width_compared_element_2/2),
                                           int(bb_center_y_compared_element_2 - bb_height_compared_element_2/2)],
                                          [int(bb_center_x_compared_element_2 + bb_width_compared_element_2/2),
                                           int(bb_center_y_compared_element_2 + bb_height_compared_element_2/2)],
                                          [int(bb_center_x_compared_element_2 - bb_width_compared_element_2/2),
                                           int(bb_center_y_compared_element_2 + bb_height_compared_element_2/2)]]
                        
                        # Compute the overlap of the two bounding boxes
                        overlap_of_boxes = calculate_iou(bbox_element_1, bbox_element_2)
                        #overlaps.append(overlap_of_boxes) # Uncomment for testing
                        
                        
                        # Remove the least confident element if there is a difference in
                        # confidences AND there is sufficient overlap between the classes.
                        # If there is not enough overlap, remove the latter of the compared
                        # elements from same_classes_in_frame and add it to cleaned_csv_data.
                        if confidence_compared_element_1 > confidence_compared_element_2:
                            if overlap_of_boxes < overlap_threshold:
                                cleaned_csv_data.append(compared_element_2)
                            same_classes_in_frame.remove(compared_element_2)
                        elif confidence_compared_element_2 > confidence_compared_element_1:
                            if overlap_of_boxes < overlap_threshold:
                                cleaned_csv_data.append(compared_element_1)
                            same_classes_in_frame.remove(compared_element_1)
                        else:
                            # A tie in confidences, so we remove the latter one
                            if overlap_of_boxes < overlap_threshold:
                                cleaned_csv_data.append(compared_element_2)
                            same_classes_in_frame.remove(compared_element_2)
                            
                    cleaned_csv_data.append(same_classes_in_frame[0])
                        
                elif len(same_classes_in_frame) == 0:
                    sys.exit('Something wrong!')
                else:
                    cleaned_csv_data.append(same_classes_in_frame[0])
    
    
    # Remove possible duplicates from the cleaned CSV (just in case, should not be any)
    cleaned_csv_data_duplicates_removed = []
    for element in cleaned_csv_data:
        if element not in cleaned_csv_data_duplicates_removed:
            cleaned_csv_data_duplicates_removed.append(element)
    
    # Write the new CSV file
    new_file = open(output_csv_file, 'w')
    new_file.close()
    with open(output_csv_file, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header_row)
        for item in cleaned_csv_data_duplicates_removed:
            writer.writerow(item)
            
    #return cleaned_csv_data_duplicates_removed, header_row, overlaps # Uncomment for testing




def map_classes(input_csv_file, output_csv_file, mapping_dict, mapping_names):
    
    """
    Perform a class mapping based on the dictionary mapping_dict.
    
    input_csv_file: The name (+ path if in another directory) of the input CSV file
    output_csv_file: The name (+ path if in another directory) of the output CSV file
    mapping_dict: A dictionary which contains the mapping information. The keys of the
                  dictionary are the desired classes, and the values of these keys are
                  lists of the original classes which are included in the desired class.
                  For example, if we have a dictionary
                  dict = {'0': ['0', '1', '2'], '1': ['4'], '2': ['3', '12']}, then we
                  are mapping the original classes '0', '1', and '2' into a new class '0',
                  we are mapping the original class '4' into a new class '1', and we are
                  mapping the original classes '3' and '12' into a new class '2'.
                  NOTE THAT if some class does not occur in the values of keys of 
                  mapping_dict, then they will be discarded from the mapping. ALSO NOTE THAT
                  a class cannot be in more than key at the same time.
    mapping_names: A dictionary which contains the same keys as in mapping_dict and the names
                   of the new classes as their values. For example,
                   dict = {'0': 'dog', '1': 'cat', '2': 'horse'}.
                  
    """
    
    # A list containing the names for the original classes for YOLOv4
    #yolo_classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 
    #                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
    #                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    #                'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
    #                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
    #                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet',
    #                'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    #                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    #                'toothbrush']
    
    # Check the consistency of mapping_dict and mapping_names
    if len(mapping_dict) != len(mapping_names):
        sys.exit('The number of elements in mapping_dict and mapping_names is not the same!')
    
    for key in mapping_dict:
        if key not in mapping_names:
            sys.exit('There are keys in mapping_dict which are not present in mapping_names!')
            
    for key in mapping_names:
        if key not in mapping_dict:
            sys.exit('There are keys in mapping_names which are not present in mapping_dict!')
    
    
    
    
    
    header_row = None
    input_csv_data = []
    
    # Read the input CSV file.
    with open(input_csv_file, 'r') as csvfile:
        csvreader_original = csv.reader(csvfile, delimiter=',')
        n = 0
        for row in csvreader_original:
            if n == 0:
                header_row = row
                n += 1
            else:
                input_csv_data.append(row)
                
    
    # First find out that which of the original classes are present in mapping_dict
    present_classes = []
    for key in mapping_dict:
        classes_in_key = mapping_dict[key]
        for item in classes_in_key:
            if item not in present_classes:
                present_classes.append(item)
    
    # Go through all the data in the CSV file and perform a mapping
    csv_data_with_mapping = []
    for row in input_csv_data:
        class_id_of_row = row[1]
        
        # If the class of the CSV row is not present in mapping_dict, then we can
        # simply skip it. Otherwise we perform a mapping for the class.
        if class_id_of_row in present_classes:
            for key in mapping_dict:
                classes_in_key = mapping_dict[key]
                for item in classes_in_key:
                    if class_id_of_row == item:
                        new_class_id = key
                        
            old_label_accuracy = row[2].split(':')[1]
            new_label_class_name = mapping_names[new_class_id]
            new_label = str(new_label_class_name) + ':' + old_label_accuracy
            csv_data_with_mapping.append([row[0], str(new_class_id), new_label, row[3], row[4], row[5], row[6], row[7]])
            
    
    # Write the new CSV file
    new_file = open(output_csv_file, 'w')
    new_file.close()
    with open(output_csv_file, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header_row)
        for item in csv_data_with_mapping:
            writer.writerow(item)




#if __name__ == '__main__':
    
    #print('Only for testing')
    # Test the bounding box cleaner function (uncomment given parts in the function for testing)
    #data, header, overlaps = bbox_cleaner('R0010861_er_csv_output_errortest.csv', 'R0010861_er_csv_output_errortest_bbox_cleaned.csv', 0.000001)
    
    # Test the mapping of the classes
    #map_dict = {'0': ['68', '69', '72'], '1': ['59']}
    #map_names = {'0': 'newclass1', '1': 'newclass2'}
    #map_classes('R01_10fps_csv_output.csv', 'test.csv', map_dict, map_names)
    
    
    
    
    
    
    
    
    
    
    
    
    