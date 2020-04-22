# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import find_peaks
from scipy.signal import welch
from datetime import datetime
from datetime import timedelta
from scipy import signal
from joblib import load

#filter used for heartrate noise reduction
def filter_hr(df):
    
    c1 = signal.medfilt(df[:],kernel_size=5)
    #n1,w1 = signal.ellipord(0.88,0.96,3,100)
    #b, a = signal.ellip(n1,3,100,w1,'low')
    #c2 = signal.lfilter(b,a,c1)
    #c2 = signal.medfilt(c2[:],kernel_size=7)
    #plt.plot(df.index,df[:],linestyle='dashed',linewidth=1)
    #plt.plot(df.index,c1,linewidth=1)
    #plt.plot(df.index,c2,color='red',linewidth=1)
    #plt.yticks(np.arange(0,200,40))
    #plt.show()
    return c1

#Fourier transform and power spectral density feature extraction adapted from http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
def get_fftpeaks(df):
    prom = 2
    t_n = 60
    N = len(df.index)
    T = t_n/N
    f_s = 1/T
    frequencies = np.linspace(0.0,1.0/(2.0*T),N//2)
    fft_df_ = fft(df[:])
    fft_df = 2.0/N * np.abs(fft_df_[0:N//2])
    promUp = 2
    promDown = 0.1
    overshootUp = False
    overshootDown = False
    fft_peaks = find_peaks(fft_df[:],prominence=prom,distance=3)
    while len(fft_peaks[0])!=6:
        if len(fft_peaks[0])>6:
            prom += promUp
            overshootUp = True
            if overshootDown:
                promDown = promDown/2
                overshootDown = False
        else: 
            prom -= promDown
            overshootDown = True
            if overshootUp:
                promUp = promUp/2
                overshootUp = False
        fft_peaks = find_peaks(fft_df[:],prominence=prom,distance=3)
    fft_peaksdf  = pd.DataFrame(fft_df[fft_peaks[0]],frequencies[fft_peaks[0]])
    return fft_peaksdf

#Power spectral density feature extraction
def get_psdpeaks(df):
    prom = 2
    t_n = 60
    N = len(df.index)
    T = t_n/N
    f_s = 1/T
    psd_df = welch(df[:],fs = f_s)
    promUp = 2
    promDown = 0.1
    overshootUp = False
    overshootDown = False
    psd_peaks = find_peaks(psd_df[:],prominence=prom,distance=5)
    while len(psd_peaks[0])!=6:
        if len(psd_peaks[0])>6:
            prom += promUp
            overshootUp = True
            if overshootDown:
                promDown = promDown/2
                overshootDown = False
        else:
            prom -= promDown
            overshootDown = True
            if overshootUp:
                promUp = promUp/2
                overshootUp = False
        psd_peaks = find_peaks(psd_df[:],prominence=prom,distance=5)
    return psd_peaks

#Jerk peak feature extraction
def get_jerkpeaks(df):
    prom = 2
    df_p = np.diff(df)
    promUp = 2
    promDown = 0.1
    overshootUp = False
    overshootDown = False
    jerk_peaks = find_peaks(df_p[:],prominence=prom,distance=5)
    while len(jerk_peaks[0])!=6:
        if len(jerk_peaks[0])>6:
            prom += promUp
            overshootUp = True
            if overshootDown:
                promDown = promDown/2
                overshootDown = False
        else:
            prom -= promDown
            overshootDown = True
            if overshootUp:
                promUp = promUp/2
                overshootUp = False
        jerk_peaks = find_peaks(df_p[:],prominence=prom,distance=5)
    jerk_peaks_y = df_p[jerk_peaks[0]]
    return jerk_peaks_y

#number of times a dataset crosses the zero-axis is determined
def get_zerocross(df):
    zeroCross = 0
    lastVal = 0
    for i in df[:]:
        if i < 0 and lastVal > 0:
            zeroCross += 1
        elif i > 0 and lastVal < 0:
            zeroCross += 1
        lastVal = i
    return zeroCross

#the mean of the input data is determined (the input is one of sensor's readings along one axis)
def get_mean(df):
    sensorMean = np.mean(df)
    return sensorMean


#classifiying determines majority activity or the first activity (in the event of no majority)
def classify(aP1,aP2,aP3):
    if aP1 == aP2:
        activity = aP1
    elif aP1 == aP3:
        activity = aP1
    elif aP2 == aP3:
        activity = aP2
    else:
        activity = aP1
    return activity

#Beginning of script file
#opens data files
acc = open('accelerometer_data_24.txt','r')
gyr = open('gyroscope_data_24.txt','r')
ppg = open('ppg_data_24.txt','r')
#creates data file for results
activities = open('activityData.txt','w')
activities.write('Date, Activity, Heart rate\n')
#importing trained model
model = load('testmodel.joblib')

acc.readline()
gyr.readline()
ppg.readline()



accdata = pd.read_table(acc,sep=",",index_col=False,names=['Time','X','Y','Z','datetime'])
gyrdata = pd.read_table(gyr,sep=",",index_col=False,names=['Time','X','Y','Z','datetime'])
ppgdata = pd.read_table(ppg,sep=",",index_col=False,names=['Time','Heartrate','datetime'])

#closing input files
acc.close()
gyr.close()
ppg.close()

del acc
del gyr
del ppg

#these statements assign the appropriate datetime for each data entry in each of the sensor readings
accdata.loc[:,'datetime'] = pd.to_datetime(accdata.loc[:,'Time'])#,'%m/%d/%y %H:%M:%S.%f')
gyrdata.loc[:,'datetime'] = pd.to_datetime(gyrdata.loc[:,'Time'])#,'%m/%d/%y %H:%M:%S.%f')
ppgdata.loc[:,'datetime'] = pd.to_datetime(ppgdata.loc[:,'Time'])

#time rounding adapted from stackoverflow by Omnifarious
#https://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object-python
initialtime = accdata.loc[0,'datetime'] - timedelta(minutes=((accdata.loc[0,'datetime']).minute%2)-2,seconds=(accdata.loc[0,'datetime']).second,microseconds=(accdata.loc[0,'datetime']).microsecond)
#initialtime = ppgdata.loc[0,'datetime'] - timedelta(minutes=((ppgdata.loc[0,'datetime']).minute%2)-2,seconds=(ppgdata.loc[0,'datetime']).second,microseconds=(ppgdata.loc[0,'datetime']).microsecond)
finaltime = accdata.loc[len(accdata.index)-1,'datetime']
#finaltime = ppgdata.loc[len(ppgdata.index)-1,'datetime']

#the datetime entries are set as the new indeces to each sensor reading
accdata = accdata.set_index(['datetime'])
gyrdata = gyrdata.set_index(['datetime'])
ppgdata = ppgdata.set_index(['datetime'])

#the length of the chunk being processed can be changed to (preferably in multiples of 60 seconds)
chunkLength = 60

timea = initialtime
timeb = initialtime+timedelta(seconds=chunkLength/4)
timec = initialtime+timedelta(seconds=chunkLength/2)
timed = initialtime+timedelta(seconds=3*chunkLength/4)
timee = initialtime+timedelta(seconds=chunkLength)
#final plan to receive data composed of 2-minunte intervals: this script reads the entry files partitioned 2-minute intervals
while (timea+timedelta(seconds=chunkLength))<finaltime:
#    each interval is partitioned into 3 chunks: each equal in time elapsed and with some overlap
#    each reading to be evaluated composed of sensor, direction, and partition (e.g. axp1 - accelerometer reading, x-direction, partition 1)
    
    axp1 = accdata.loc[timea:timec,'X']
    axp2 = accdata.loc[timeb:timed,'X']
    axp3 = accdata.loc[timec:timee,'X']
    ayp1 = accdata.loc[timea:timec,'Y']
    ayp2 = accdata.loc[timeb:timed,'Y']
    ayp3 = accdata.loc[timec:timee,'Y']
    azp1 = accdata.loc[timea:timec,'Z']
    azp2 = accdata.loc[timeb:timed,'Z']
    azp3 = accdata.loc[timec:timee,'Z']
    
    gxp1 = gyrdata.loc[timea:timec,'X']
    gxp2 = gyrdata.loc[timeb:timed,'X']
    gxp3 = gyrdata.loc[timec:timee,'X']
    gyp1 = gyrdata.loc[timea:timec,'Y']
    gyp2 = gyrdata.loc[timeb:timed,'Y']
    gyp3 = gyrdata.loc[timec:timee,'Y']
    gzp1 = gyrdata.loc[timea:timec,'Z']
    gzp2 = gyrdata.loc[timeb:timed,'Z']
    gzp3 = gyrdata.loc[timec:timee,'Z']
    
#ppg reading is only sensor reading to not be manipulated for feature extraction
    ppg1 = ppgdata.loc[timea:timee,'Heartrate']
    partitionHeart= np.mean(ppg1)
    
    
#all relevant features (in this case jerk peaks) are extracted    
    axp1_jerk = np.sort(get_jerkpeaks(axp1))[::-1]
    axp2_jerk = np.sort(get_jerkpeaks(axp2))[::-1]
    axp3_jerk = np.sort(get_jerkpeaks(axp3))[::-1]
    ayp1_jerk = np.sort(get_jerkpeaks(ayp1))[::-1]
    ayp2_jerk = np.sort(get_jerkpeaks(ayp2))[::-1]
    ayp3_jerk = np.sort(get_jerkpeaks(ayp3))[::-1]
    azp1_jerk = np.sort(get_jerkpeaks(azp1))[::-1]
    azp2_jerk = np.sort(get_jerkpeaks(azp2))[::-1]
    azp3_jerk = np.sort(get_jerkpeaks(axp3))[::-1]
    
    gxp1_jerk = np.sort(get_jerkpeaks(gxp1))[::-1]
    gxp2_jerk = np.sort(get_jerkpeaks(gxp2))[::-1]
    gxp3_jerk = np.sort(get_jerkpeaks(gxp3))[::-1]
    gyp1_jerk = np.sort(get_jerkpeaks(gyp1))[::-1]
    gyp2_jerk = np.sort(get_jerkpeaks(gyp2))[::-1]
    gyp3_jerk = np.sort(get_jerkpeaks(gyp3))[::-1]
    gzp1_jerk = np.sort(get_jerkpeaks(gzp1))[::-1]
    gzp2_jerk = np.sort(get_jerkpeaks(gzp2))[::-1]
    gzp3_jerk = np.sort(get_jerkpeaks(gxp3))[::-1]
    
#all features for each partition are concatenated to be fed simultaneously to classifying model    
    p1_jerk = np.concatenate((axp1_jerk,ayp1_jerk,azp1_jerk,gxp1_jerk,gyp1_jerk,gzp1_jerk))
    p2_jerk = np.concatenate((axp2_jerk,ayp2_jerk,azp2_jerk,gxp2_jerk,gyp2_jerk,gzp2_jerk))
    p3_jerk = np.concatenate((axp3_jerk,ayp3_jerk,azp3_jerk,gxp3_jerk,gyp3_jerk,gzp3_jerk))    
    
#predictions to each partition are made using the features
    pred_par1 = model.predict(p1_jerk.reshape(1,-1))
    pred_par2 = model.predict(p2_jerk.reshape(1,-1))
    pred_par3 = model.predict(p3_jerk.reshape(1,-1))
    
#the activity picked is either    
    partitionActivity = classify(pred_par1,pred_par2,pred_par3)
    partitionTime = timea.strftime('%m/%d/%y %H:%M:%S')
    
#writes findings from data to output file    
    fileLine = partitionTime+", "+partitionActivity[0]+", "+str(partitionHeart)+"\n"
    activities.write(fileLine)
    
    #incrementing times for new chunk
    timea += timedelta(seconds=chunkLength)
    timeb += timedelta(seconds=chunkLength)
    timec += timedelta(seconds=chunkLength)
    timed += timedelta(seconds=chunkLength)
    timee += timedelta(seconds=chunkLength)

#closing output file
activities.close()
