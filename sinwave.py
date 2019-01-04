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
sliding_window = [] #this traverses the entire signal
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
        
    
input_dim=2
last_value=array([0 for i in range (input_dim)]) #array containing 0
last_derivative=array([0 for i in range (input_dim)])


'''
if the input at time t is x(t), the derivative is x'(t) = (x(t) – x(t-1)). 
Following the analogy, x”(t) = (x'(t) – x'(t-1)). 
Here’s the code for that:
'''
def get_total_io():
    '''
    Returns the overall Input and Output as required by the model.
    The input is a concatenation of the wave values, their first and
    second derivatives
    '''
    global last_value,last_derivative
    row_i,row_o=get_pair()
    row_i=row_i[0]
    l1=list(row_i)
    derivative=row_i -last_value
    l2=list(derivative)
    last_value=row_i
    l3 = list(derivative - last_derivative)
    last_derivative = derivative
    return array([l1 + l2 + l3]), row_o

    
    '''
    upto this,the code looks like it computes the derivatives and slopes
    '''
    
import tensorflow as tf
from tensorflow.nn import *
##The Input Layer as a Placeholder
#Since we will provide data sequentially, the 'batch size'
#is 1.
'''
A placeholder is simply a variable that we will assign data to at a later date. 
It allows us to create our operations and build our computation graph, 
without needing the data. In TensorFlow terminology,
 we then feed data into the graph through these placeholders.
'''
input_layer=tf.placeholder(tf.float32,[1,input_dim*3])
#now ,we are using the lstm (long short term layer)
lstm_layer1=rnn_cell.BasicLSTMCell(input_dim*3)


   


