import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import datetime as dt

def plotEstimates(kayakDataFrame,savepath=None):
    for channel in set(kayakDataFrame.marketing_channel):
        singleChannelKayakDataFrame = kayakDataFrame[(kayakDataFrame.marketing_channel == channel)]
        fig = plt.figure(figsize=(15, 12))
        uniqueContries = set(singleChannelKayakDataFrame.country_code)
        for i, country in enumerate(uniqueContries):
            plt.subplot(len(uniqueContries),1,i+1)
            if i == 0:
                plt.title(singleChannelKayakDataFrame.keys()[-1] + ' for ' + channel)
            thisKayakDataFrame = singleChannelKayakDataFrame[singleChannelKayakDataFrame.country_code == country]
            #del thisKayakDataFrame['marketing_channel']
            #del thisKayakDataFrame['country_code']

            #check if there is a low number samples
            if thisKayakDataFrame.shape[0] < 150:
                print country + ' ' + channel
                thisKayakDataFrame = replaceMissingDaysWithZero(thisKayakDataFrame)
                endMonthMedian = thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].loc[pd.date_range(dt.datetime(2015,8,01,00,00,00), periods=31, freq='D')].median()
                earlierMonthMedian = thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].loc[pd.date_range(dt.datetime(2015,4,01,00,00,00), periods=31, freq='D')].median()
                if endMonthMedian > 3.0*earlierMonthMedian:
                    #then assum trending
                    print ' trending'
                    plt.plot(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].values,'r')
                    plt.ylabel(country)
                else:
                    plt.plot(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].values,'k')
                    plt.ylabel(country)
                continue

            #handle cases differently when there is a low overall number
            if thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].median() < 100:
                print country + ' ' + channel + ' median of ' + str(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].median()) + ' std of ' + str(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].std())
                plt.subplot(len(uniqueContries),1,i+1)
                plt.plot(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].values)
                plt.plot(np.array([1,thisKayakDataFrame.shape[0]]),thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].mean()*np.ones(2),'k',linewidth = 2)
                plt.ylabel(country)
                continue

            #check for data integrety
            thisKayakDataFrame = replaceMissingDaysWithZero(thisKayakDataFrame)

            #check weekly power ratio
            powRatio = plotFourrie(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].values,plots = 0)

            if powRatio > 0.8:
                copyKayakDataFrame = thisKayakDataFrame.copy('deep')
                del copyKayakDataFrame['marketing_channel']
                del copyKayakDataFrame['country_code']
                trended = trendDecomposition(copyKayakDataFrame)
                appended = appendTimeProxi(trended)
                plt.subplot(len(uniqueContries),1,i+1)
                plt.plot(appended['trend'].values)
                plt.plot(appended[appended.keys()[0]].values)
                plt.ylabel(country)
            else:
                plt.subplot(len(uniqueContries),1,i+1)
                plt.plot(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].values)
                plt.ylabel(country)

        if savepath:
            #plt.savefig(savepath + channel + '.jpg')
            #plt.close()
            pass

def plotFourrie(npArray,plots = 1):
    samples2use = int(2**np.floor(np.log2(npArray.shape[0])))
    fftVals = np.fft.fft(npArray.astype(int),n=samples2use,axis=-1)
    #fftVals = np.fft.fft(npArray.astype(int))
    power = np.abs(fftVals)**2
    freqs = np.fft.fftfreq(samples2use, 1)
    if plots:
        plt.figure()
        plt.plot(np.fft.fftshift(freqs),np.fft.fftshift(power))

    freqs = np.fft.fftshift(freqs)
    power = np.fft.fftshift(power)
    weekpeak = power[np.where(freqs < 1.0/6.0) and np.where(freqs > 1.0/8.0)].max()
    globalpeak= power[np.where(freqs < 1.0/2.5) and np.where(freqs > 1.0/30)].max()
    return weekpeak/globalpeak

def trendDecomposition(kayakDF2detrend):
    zeroidx = np.where(kayakDF2detrend[kayakDF2detrend.keys()[-1]].values == 0)[0]
    kayakDF2detrend[kayakDF2detrend.keys()[-1]][zeroidx] = np.finfo(float).eps
    #weekly decomposition
    import statsmodels.api as sm
    res = sm.tsa.seasonal_decompose(kayakDF2detrend.values,model="multiplicative",freq=7)
    kayakDF2detrend['trend'] = res.trend
    return kayakDF2detrend

def replaceMissingDaysWithZero(kayakDataFrameWithMissing):
    # like many functions used here, I take advantage of the fact that in these
    # data frames only the last key has the numberical value of interest
    # I use this to generalize my function for conversions and visits
    key2Edit = kayakDataFrameWithMissing.keys()[-1]

    #make sure there is a time index for a consistant range
    idx = pd.date_range(dt.datetime(2014,11,01,00,00,00),dt.datetime(2015,8,31,00,00,00),freq = 'D')
    kayakRedex = kayakDataFrameWithMissing.reindex(idx)

    #the assumption here is that the data is missing because it is 0 so repace Nan values with epsilon
    #epsilon is used here for numerical stability in later calculations
    kayakRedex[key2Edit][np.where(np.isnan(kayakRedex[key2Edit]))[0]] = np.finfo(float).eps

    return kayakRedex

def appendTimeProxi(dFrame2append):
    #adding as missing data
    #extending Nov and Dec as from year 2014

    base = dt.datetime(2015,9,01,00,00,00)
    septToOct = [base + dt.timedelta(days=x) for x in range(0, 61)]
    for date in septToOct:
        dFrame2append.loc[date] = np.nan

    #taggin on nov and dec 2014 as proxi for 2015
    #rng1 = pd.date_range(dt.datetime(septToOct[-1], periods=61, freq='D')
    #rng2 = pd.date_range(dt.datetime(2014,11,01,00,00,00), periods=61, freq='D')
    for day in np.arange(1, 62):
        dFrame2append.loc[septToOct[-1] + dt.timedelta(days=day)] = dFrame2append.loc[dt.datetime(2014,11,01,00,00,00) + dt.timedelta(days=day-1)]

    return dFrame2append


#conversions = pd.read_csv('./data_files/conversions.csv',index_col = 'datestamp',parse_dates=True)
conversions = pd.read_csv('D:/KAYAKtest/data_files/conversions.csv',index_col = 'datestamp',parse_dates=True)

plotEstimates(conversions)

