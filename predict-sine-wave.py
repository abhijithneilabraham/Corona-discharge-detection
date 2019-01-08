import os
import sys

# 0, 1, or 2 How much do you want to know if using TF?
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#Force a Keras backend - Pick one and only one, Vasili
os.environ['KERAS_BACKEND']='tensorflow'
#os.environ['KERAS_BACKEND']='cntk'

import pandas as pd
import numpy as np
import keras as ks
import tensorflow as tf
#import cntk as c
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import losses

np.random.seed(70)
back_end = ks.backend.backend()

#got to chart the prediction, 'labels' for those in Lutz, Fl.
import matplotlib
#http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
#matplotlib.use('Agg') #uncomment this if using on a web service with no grahics screen, ie headless.
import matplotlib.pyplot as plt

#Last Tested With:
#Python version: 3.6.2 
#Numpy version: 1.13.1
#Matplotlib version: 2.0.2
#Keras version: 2.0.6
#Tensorflow version: 1.1.0
#CNTK version: 2.1 GPU with 1bit-SGD


# Lets know what versions we are working with now.
print("\n") #newline
print("Python version: {}".format(sys.version))
print("Numpy version: {}".format(np.version.version))
print("Matplotlib version: {}".format(matplotlib.__version__))
print("Keras version: {}".format(ks.__version__))
print("Tensorflow version: {}".format(tf.__version__))
#print("CNTK version: {}".format(c.__version__))
print("\n") #newline


##########################  Create Sine Waves Data Frame ############################
#only certain combinations will result in all 4 waves being generated correctly.
#odd numbers for f, best to have Fs and sample the same.
frequencey = 2500
sample = 2500
f = 3

z = np.arange(sample)
zero = np.sin(2 * np.pi * f * z / frequencey)
zero = (zero - np.max(zero))/-np.ptp(zero)
ninty = np.sin(2 * np.pi * f * (z+(sample*.25)) / frequencey)
ninty = (ninty - np.max(ninty))/-np.ptp(ninty)
oneEighty = np.sin(2 * np.pi * f * (z+(sample*.5)) / frequencey)
oneEighty = (oneEighty - np.max(oneEighty))/-np.ptp(oneEighty)
twoSeventy = np.sin(2 * np.pi * f * (z+(sample*.75)) / frequencey)
twoSeventy = (twoSeventy - np.max(twoSeventy))/-np.ptp(twoSeventy)

df = pd.DataFrame({ "ind":np.array(range(0,sample,1)),
                    "Zero":zero, 
                    "Ninty":ninty, 
                    "One Eighty":oneEighty, 
                    "Two Seventy":twoSeventy})

# convert an array of values into a df matrix
def create_dataset(dataset, look_back=1):
	dataX = []
	for i in range(look_back,len(dataset)+1):
		a = dataset[(i-look_back):(i), 0]
		dataX.append(a)
	return np.array(dataX)


look_back = 10

zero = df.loc[:, lambda df: ['Zero']].values
zero = create_dataset(zero,look_back)
ninety = df.loc[:, lambda df: ['Ninty']].values
ninety = create_dataset(ninety,look_back)
oneEighty = df.loc[:, lambda df: ['One Eighty']].values
oneEighty = create_dataset(oneEighty,look_back)


################## Get Data Ready for Processing #################
X = np.stack((zero,ninety,oneEighty),axis=1)
X = X.swapaxes(1,2)

Y = df.loc[look_back-1:,lambda df: ['Two Seventy']].values

# how much to use for training, how much to hold back for testing.
dataSplit = .5 # two thirds sounds right? Mark of the devil?

train_size = int(len(X) * dataSplit)
test_size = len(X) - train_size

train_X= X[0:train_size,:].astype('float32')
test_X = X[train_size-1:len(X),:].astype('float32')
train_Y= Y[0:train_size,:].astype('float32')
test_Y = Y[train_size-1:len(Y),:].astype('float32')


# https://stackoverflow.com/questions/26778079/valueerror-ndarray-is-not-c-contiguous-in-cython
if back_end.upper() == "CNTK":
    train_X = train_X.copy(order='C')
    test_X = test_X .copy(order='C')
    train_Y = train_Y.copy(order='C')
    test_Y = test_Y.copy(order='C')

################## Create Model #################

model = Sequential()
output_space = 6 #(minimum 2 times the input, would that be right for nyquist sample rates? 3 inputs times 2?)
epochs = 50
batch_size = 10
stop_delta = .001
stop_patience = 5
drop_out = .1
gaussian_noise = 0 # Doesnt work yet with CNTK Keras backend.

# 'GRU' or 'LSTM'
# GRU is relatively new, it's performance is on par with LSTM, computationally more efficient.

model_type = ks.layers.recurrent.LSTM.__name__

# Available activations softmax elu softplus softsign relu tanh sigmoid linear 
rnn_activation = ks.activations.linear.__name__
DropOutActType = ks.activations.linear.__name__

if model_type == "GRU":
    from keras.layers import GRU
    model.add(GRU(units=output_space, input_shape=(train_X.shape[1],train_X.shape[2]), 
                  activation=rnn_activation,
                  kernel_initializer=ks.initializers.ones.__name__, 
                  bias_initializer=ks.initializers.ones.__name__,
                  kernel_regularizer=None,
                  bias_regularizer=None, 
                  activity_regularizer=None, 
                  kernel_constraint=None, 
                  bias_constraint=ks.constraints.max_norm(1)))
 
else:
    from keras.layers import LSTM
    model.add(LSTM(units=output_space, input_shape=(train_X.shape[1],
              train_X.shape[2]), activation=rnn_activation))
    

# Add drop out if its set above 0
if drop_out > 0:
        from keras.layers import Dropout
        model.add(Dropout(drop_out))
        
if gaussian_noise > 0:
        from keras.layers import GaussianNoise
        model.add(GaussianNoise(2))

model.add(Dense(1))

loss_func =  ks.losses.logcosh.__name__
opt_func = str(ks.optimizers.Adam.__name__)

################## Compile and Fit Model #################
model.compile(loss=loss_func,optimizer=opt_func,sample_weight_mode='none')
early_stopping = [EarlyStopping(monitor='loss',patience=stop_patience,
                 verbose=1,min_delta=stop_delta,mode='auto')]

#verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
#initial_epoch: epoch at which to start training (useful for resuming a previous training run)
ho = model.fit(train_X, train_Y, epochs=epochs,shuffle=True, batch_size=batch_size, 
               verbose=1,callbacks=early_stopping,initial_epoch=0)

epoch_count = len(ho.epoch)

if epoch_count < epochs:
    print("\n")
    print ("Early stop. Completed " + str(epoch_count) + " epochs of " + str(epochs))
    print("\n")


# make predictions
trainPredict = model.predict(train_X)
testPredict = model.predict(test_X)

testPredictplot = np.insert(testPredict,0,np.repeat(np.nan,
                     len(trainPredict) -1))

trainPredictplot = np.insert(trainPredict,len(trainPredict),
                             np.repeat(np.nan,len(testPredict)))

################## Lets See A Chart #################

chart_title="Multiple Sine Wave Prediction"

# plot baseline and predictions
plt.rc("font", size=12)
plt.rc("figure",figsize = (12,6))
back_end = ks.backend.backend()
plt.title(chart_title + ' - ' + "Flavor:" + model_type + " " + "Activation:" + rnn_activation.upper() 
          + " " + "Back End:" + back_end.upper())

plt.plot(zero[:,look_back-1], color='yellow', alpha=0.40, label='0° Phase Sine')
plt.plot(ninety[:,look_back-1], color='green', alpha=0.1, label='90° Phase Sine') 
plt.plot(oneEighty[:,look_back-1], color='blue', alpha=0.1, label='180° Phase Sine') 
plt.plot(trainPredictplot, color='black', alpha=0.4, label='270° Training')
plt.plot(testPredictplot, color='red', alpha=0.8, label='270° Predicted')
plt.legend(loc='upper left')

cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_visible(False)
cur_axes.axes.get_yaxis().set_visible(True)

plt.figtext(0.51, 0.05, str(epoch_count) + ' Of ' +  str(epochs) + ' Epochs -' + 
            ' Batch Size: ' + str(batch_size) + " - " +
            ' Drop Out: ' + str(round(drop_out*100,1)) + "%"  + " - " +
            ' Look Back: ' + str(look_back) +  " - " +
            ' Train/Test Split: ' + str(round(dataSplit*100,1)) + "%",
            horizontalalignment='center')

#try:
#    os.remove("PredictSineWave.png")
#except OSError:
#    pass

# You want to see or save? No head, then comment out the show and keep the image save.
#plt.savefig("PredictSineWave.png",dpi=150,bbox_inches=None,pad_inches = 0,frameon=None)
plt.show() # comment out this line if headless