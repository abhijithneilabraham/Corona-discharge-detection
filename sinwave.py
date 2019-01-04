#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:28:05 2019

@author: abhijithneilabraham
"""

from numpy import array,sin,cos,pi
from random import random
'''
choosing random initial values of angles 
so,this value of angles is what we change according to the training dataset 
and then try to match it together,by training using lstm.
'''
angle1=random()
angle2=random()
frequency1=100 #frequency here is set that a period of 2pi is subdivided into several periods
frequency2=200
lag=23 #this defines how many steps ahead we are trying to predict
def get_sample():
    #this function is gonna return a sin and  cos value 
    global angle1,angle2
    angle1+=2*pi/float(frequency1) #sampling the angles in the frequency period in the 0-2pi range
    angle2+=2*pi/float(frequency2)
    angle1 %=2*pi #after sampling the angles,taking the value of angle to be converted to sin and cos waves
    angle2 %=2*pi
    return array([array([
            5+5*sin(angle1)+10*cos(angle2),
            7+7*sin(angle2) + 14*cos(angle1)
                      ])])
sliding_window = []
for i in range(lag-1):
    sliding_window.append(get_sample())

def get_pair():
    '''
    returns a pair,with input and output
    the output pair will be i steps ahead of the input 
    pair,as per the frequency
    '''
    global sliding_window
    sliding_window.append(get_sample())
    input_value=sliding_window[0]
    output_value=sliding_window[-1]
    sliding_window=sliding_window[1:]
    return input_value,output_value
        
    
    


