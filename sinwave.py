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
frequency1=100 #frequency range in range of 50-60 is chosen as standard.
frequency2=200
lag=23 #this defines how many steps ahead we are trying to predict
def get_sample():
    #this function is gonna return a sin or cos value 
    global angle1,angle2
    angle1+=2*pi/float(frequency1) #sampling the angles in the frequency period in
    angle2+=2*pi/float(frequency2)
    angle1 %=2*pi
    angle2 %=2*pi
    


