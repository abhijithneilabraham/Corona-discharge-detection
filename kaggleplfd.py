import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pyarrow.parquet as pq

print(os.listdir("../input")) #just listing the files that I have
train = pq.read_pandas('../input/train.parquet').to_pandas()#loading the training dataset.It is in apache parquet format
import matplotlib.pyplot as plt #plotting some patterns.
train.iloc[:, 0:3].plot(figsize=(12, 8))#iloc selects row and columns by number.figsize plots the figures using matplotlib
plt.axis('off');#turns of the axis
df=df = pd.read_parquet('../input/train.parquet')
df.to_csv('../input/training.csv')
from scipy.fftpack import fft

