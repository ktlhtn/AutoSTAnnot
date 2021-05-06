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
