# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
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

def filter_hr(df):
    
    c1 = signal.medfilt(df[:],kernel_size=5)
    n1,w1 = signal.ellipord(0.88,0.96,3,100)
    b, a = signal.ellip(n1,3,100,w1,'low')
    c2 = signal.lfilter(b,a,c1)
    c2 = signal.medfilt(c2[:],kernel_size=7)
    plt.plot(df.index,df[:],linestyle='dashed',linewidth=1)
    plt.plot(df.index,c1,linewidth=1)
    plt.plot(df.index,c2,color='red',linewidth=1)
    plt.yticks(np.arange(0,200,40))
    plt.show()
    return c2

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
    fft_peaks = find_peaks(fft_df[:],prominence=prom,distance=3)
    while len(fft_peaks[0])!=6:
        if len(fft_peaks[0])>6:
            prom += 1
        else: 
            prom -= 0.0025
        fft_peaks = find_peaks(fft_df[:],prominence=prom,distance=3)
    fft_peaksdf  = pd.DataFrame(fft_df[fft_peaks[0]],frequencies[fft_peaks[0]])
    return fft_peaksdf

def get_psdpeaks(df):
    prom = 2
    t_n = 60
    N = len(df.index)
    T = t_n/N
    f_s = 1/T
    psd_df = welch(df[:],fs = f_s)
    plt.plot(psd_df[0],psd_df[1])
    psd_peaks = find_peaks(psd_df[:],prominence=prom,distance=5)
    while len(psd_peaks[0])!=6:
        if len(psd_peaks[0])>6:
            prom += prom/2
        else:
            prom -= prom/2
        psd_peaks = find_peaks(psd_df[:],prominence=prom,distance=5)
    return psd_peaks

def get_jerkpeaks(df):
    prom = 2
    df_p = np.diff(df)
    jerk_peaks = find_peaks(df_p[:],prominence=prom,distance=5)
    while len(jerk_peaks[0])!=6:
        if len(jerk_peaks[0])>6:
            prom += 1
        else:
            prom -= 0.0025
        jerk_peaks = find_peaks(df_p[:],prominence=prom,distance=5)
    jerk_peaks_y = df_p[jerk_peaks[0]]
    return jerk_peaks_y

input('Hold onto your butts')

acc = open('accelerometer_data_24_3.txt','r')
gyr = open('gyroscope_data_24_3.txt','r')
ppg = open('ppg_data_24_1.txt','r')
activities = open('activityData.txt','w')
#ppg = open('ppg_data_24.txt','r')

model = load('testmodel.joblib')

acc.readline()
gyr.readline()
ppg.readline()



accdata = pd.read_table(acc,sep=",",index_col=False,names=['Time','X','Y','Z','datetime'])
gyrdata = pd.read_table(gyr,sep=",",index_col=False,names=['Time','X','Y','Z','datetime'])
ppgdata = pd.read_table(ppg,sep=",",index_col=False,names=['Time','Heartrate','datetime'])


#these statements assign the appropriate datetime for each data entry in each of the sensor readings
accdata.loc[:,'datetime'] = pd.to_datetime(accdata.loc[:,'Time'])#,'%m/%d/%y %H:%M:%S.%f')
gyrdata.loc[:,'datetime'] = pd.to_datetime(gyrdata.loc[:,'Time'])#,'%m/%d/%y %H:%M:%S.%f')
ppgdata.loc[:,'datetime'] = pd.to_datetime(ppgdata.loc[:,'Time'])
#for i in range(0,len(ppgdata.index)):
#    ppgdata.loc[i,'datetime'] = datetime.strptime(ppgdata.loc[i,'Time'],'%m/%d/%y %H:%M:%S.%f')

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

timea = initialtime
timeb = initialtime+timedelta(seconds=30)
timec = initialtime+timedelta(seconds=60)
timed = initialtime+timedelta(seconds=90)
timee = initialtime+timedelta(seconds=120)
#final plan to receive data composed of 2-minunte intervals: this script reads the entry files partitioned 2-minute intervals
while (timea+timedelta(minutes=2))<finaltime:
#    each 2-minute interval is partitioned into 3 chunks: each equal in time elapsed and with some overlap
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
    
#    ppg reading only sensor reading to not be manipulated for feature extraction
    ppg1 = ppgdata.loc[timea:timea+timedelta(minutes=2),'Heartrate']
    ppgm= np.mean(ppg1)
#    
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
    
    p1_jerk = np.concatenate((axp1_jerk,ayp1_jerk,azp1_jerk,gxp1_jerk,gyp1_jerk,gzp1_jerk))
    p2_jerk = np.concatenate((axp2_jerk,ayp2_jerk,azp2_jerk,gxp2_jerk,gyp2_jerk,gzp2_jerk))
    p3_jerk = np.concatenate((axp3_jerk,ayp3_jerk,azp3_jerk,gxp3_jerk,gyp3_jerk,gzp3_jerk))    

    pred_par1 = model.predict(p1_jerk.reshape(1,-1))
    pred_par2 = model.predict(p2_jerk.reshape(1,-1))
    pred_par3 = model.predict(p3_jerk.reshape(1,-1))
    
#    pred_array = np.concatenate(pred_par1,pred_par2,pred_par3)
    
    
#    activities.write()
    
    timea += timedelta(minutes=2)
    timeb += timedelta(minutes=2)
    timec += timedelta(minutes=2)
    timed += timedelta(minutes=2)
    timee += timedelta(minutes=2)


acc.close()
gyr.close()
ppg.close()