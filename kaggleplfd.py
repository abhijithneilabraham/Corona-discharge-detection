import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input")) #just listing the files that I have
train = pd.read_parquet("../input/train.parquet") #loading the training dataset.It is in apache parquet format
import matplotlib.pyplot as plt #plotting some patterns.
train.iloc[:, 0:3].plot(figsize=(12, 8))#iloc selects row and columns by number.figsize plots the figures using matplotlib
plt.axis('off');#turns of the axis
import tensorflow as tf
fourier=tf.spectral.fft(train.iloc[:,0:3])
