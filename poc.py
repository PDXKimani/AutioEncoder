import wave, struct
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
import tensorflow as tf
import numpy as np
from functools import reduce
import sys

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)

def dataFromWave(fname):
    """ return list with interleaved samples """
    f = wave.open(fname, 'rb')
    chans = f.getnchannels()
    samps = f.getnframes()
    sampwidth = f.getsampwidth()
    rate = f.getframerate()
    if  sampwidth == 3: #have to read this one sample at a time
        s = ''
        for k in xrange(samps):
            fr = f.readframes(1)
            for c in xrange(0,3*chans,3):                
                s += '\0'+fr[c:(c+3)] # put TRAILING 0 to make 32-bit (file is little-endian)
    else:
        s = f.readframes(samps)
    f.close()
    unpstr = '<{0}{1}'.format(samps*chans, {1:'b',2:'h',3:'i',4:'i',8:'q'}[sampwidth])
    x = list(struct.unpack(unpstr, s))
    if sampwidth == 3:
        x = [k >> 8 for k in x] #downshift to get +/- 2^24 with sign extension
    return x, chans, samps, sampwidth, rate

def dataToWave(fname, data, chans, samps, sampwidth, rate):
    obj = wave.open(fname,'wb')
    obj.setnchannels(chans)
    obj.setsampwidth(width)
    obj.setframerate(rate)
    packstr = "<{0}".format({1:'b',2:'h',3:'i',4:'i',8:'q'}[sampwidth])
    for i in range(samps*chans):
        obj.writeframesraw(struct.pack(packstr, data[i]))
    obj.close()

data, chans, samps, width, samp_rate = dataFromWave("files/01.wav")

data = np.array(data)

rate = samp_rate//100
data = data.astype(float) / float(pow(2,15))
data += 1.0
data = data / 2.0

n_in = len(data)
p_size = n_in + (rate - (n_in % rate))
padded = np.zeros((p_size,))
padded[0:n_in] = data

chunks = np.split(padded, p_size//(rate))

inputs = np.array(chunks)

model = Sequential()

model.add(Dense(rate, input_shape=(rate,),activation='relu'))
model.add(Dense(rate//2, activation='relu'))
model.add(Dense(rate//4, activation='relu'))
model.add(Dense(rate//2, activation='relu'))
model.add(Dense(rate, activation='sigmoid'))

model.compile(optimizer='adam', loss='mse')

model.fit(inputs, inputs, epochs=10)

outputs = model.predict(inputs)

out = np.concatenate(outputs)

out = (((out * 2.0) - 1.0) * float(pow(2,15))).astype(int)

def norm(x):
    if x < -32768:
        return 0
    if x > 32767:
        return 0
    return x
out = list(map(norm, out))

dataToWave("out-foo11.wav", out, chans, samps, width, samp_rate)