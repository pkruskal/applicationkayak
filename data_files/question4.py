import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import datetime as dt
import scipy as sci

def removeOutlier4Conversions(conversions):
    #conversions.loc[conversions['conversions'] > 100000] = np.nan
    conversions = conversions.loc[conversions['conversions'] < 100000]
    return conversions

def findRate():
    #conversions = pd.read_csv('D:/KAYAKtest/data_files/conversions.csv',index_col = 'datestamp',parse_dates=True)
    #visits = pd.read_csv('D:/KAYAKtest/data_files/visits.csv',index_col = 'datestamp',parse_dates=True)
    conversions = pd.read_csv('E:/applications/KAYAK/data_files/conversions.csv',index_col = 'datestamp',parse_dates=True)
    visits = pd.read_csv('E:/applications/KAYAK/data_files/visits.csv',index_col = 'datestamp',parse_dates=True)

    #remove outlier
    conversions = removeOutlier4Conversions(conversions)

    #merge them
    conversions.reset_index(level=0, inplace=True)
    visits.reset_index(level=0, inplace=True)
    visitsAndConversions = pd.merge(visits, conversions, how='outer', on=['datestamp', 'country_code','marketing_channel'])
    visitsAndConversions = visitsAndConversions.set_index('datestamp')
    #if there is a visit value, but no conversion value (ie nan) then set conversion to 0
    visitsAndConversions.conversions.values[[[np.array(~np.isnan(visitsAndConversions.user_visits)) * np.array(np.isnan(visitsAndConversions.conversions))][0]]] = 0
    visitsAndConversions['rate'] = visitsAndConversions['conversions']/visitsAndConversions['user_visits']
    return visitsAndConversions

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
    globalmean= power[np.where(freqs < 1.0/2.5) and np.where(freqs > 1.0/30)].mean()
    globalstd = power[np.where(freqs < 1.0/2.5) and np.where(freqs > 1.0/30)].std()
    weeklyPowerStrength = (weekpeak-globalmean)/globalstd
    return weeklyPowerStrength

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

def trendDecomposition(kayakDF2detrend,channel2detrend = None):
    import statsmodels.api as sm

    if channel2detrend == None:
        channel2detrend = kayakDF2detrend.keys()[-1]

    #for numerical reasons the we can have no 0's for the multiplicative detrending model
    zeroidx = np.where(kayakDF2detrend[channel2detrend].values == 0)[0]
    kayakDF2detrend[kayakDF2detrend.keys()[-1]][zeroidx] = np.finfo(float).eps

    #weekly decomposition
    res = sm.tsa.seasonal_decompose(kayakDF2detrend.values,model="multiplicative",freq=7)
    kayakDF2detrend['trend'] = res.trend
    kayakDF2detrend['seasonal'] = res.seasonal+res.trend
    return kayakDF2detrend

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

def plotRateEstimates(kayakDataFrame,savepath=None):
    channelRates = {}

    for channel in set(kayakDataFrame.marketing_channel):
        channelRates[channel] = []

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

            print country + ' ' + channel

            #check if there is a low number samples
            if thisKayakDataFrame.shape[0] < 200:
                thisKayakDataFrame = replaceMissingDaysWithZero(thisKayakDataFrame)
                thisKayakDataFrame = replaceMissingDaysWithZero(thisKayakDataFrame)
                channelRates[channel].append(thisKayakDataFrame['rate'].mean())


                endMonthMedian = thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].loc[pd.date_range(dt.datetime(2015,8,01,00,00,00), periods=31, freq='D')].median()
                earlierMonthMedian = thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].loc[pd.date_range(dt.datetime(2015,4,01,00,00,00), periods=31, freq='D')].median()
                if endMonthMedian > 3.0*earlierMonthMedian:
                    #then assum trending
                    print ' trending'
                    appended = appendTimeProxi(thisKayakDataFrame)
                    plt.plot(appended[appended.keys()[-1]].values,'r')
                    plt.ylabel(country)
                else:
                    appended = appendTimeProxi(thisKayakDataFrame)
                    plt.plot(appended[appended.keys()[-1]].values,'k')
                    plt.ylabel(country)
                continue

            #replace missing data with epsilon values
            thisKayakDataFrame = replaceMissingDaysWithZero(thisKayakDataFrame)


            '''
            #handle cases differently when there is a low overall number
            if thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].median() < 100:
                print country + ' ' + channel + ' median of ' + str(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].median()) + ' std of ' + str(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].std())
                appended = appendTimeProxi(thisKayakDataFrame)
                plt.subplot(len(uniqueContries),1,i+1)
                plt.plot(appended[appended.keys()[-1]].values)
                plt.plot(np.array([1,appended.shape[0]]),appended[appended.keys()[-1]].mean()*np.ones(2),'k',linewidth = 2)
                plt.ylabel(country)
                continue
            '''

            thisKayakDataFrame['rollingMeanRate'] = pd.rolling_mean(thisKayakDataFrame['rate'],14)
            #nanIdx = np.where(np.isnan(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].values))[0]
            #thisKayakDataFrameNoNans = thisKayakDataFrame.dropna(subset = ['rollingMeanRate'])

            appended = appendTimeProxi(thisKayakDataFrame)
            appended = appended.interpolate()
            plt.subplot(len(uniqueContries),1,i+1)
            plt.plot(appended['rollingMeanRate'].values)
            plt.ylabel(country)

            channelRates[channel].append(appended[thisKayakDataFrame.keys()[-1]].loc[pd.date_range(dt.datetime(2015,9,01,00,00,00), periods=30, freq='D')].mean())

        print channel
        if savepath:
            plt.savefig(savepath + str(channel) + '.jpg')
            plt.close()
    return channelRates

def plotVisitsEstimates(kayakDataFrame,savepath=None):
    channelVisitsEstimate = {}

    for channel in set(kayakDataFrame.marketing_channel):
        channelVisitsEstimate[channel] = []

        singleChannelKayakDataFrame = kayakDataFrame[(kayakDataFrame.marketing_channel == channel)]
        fig = plt.figure(figsize=(15, 12))
        uniqueContries = set(singleChannelKayakDataFrame.country_code)
        for i, country in enumerate(uniqueContries):
            plt.subplot(len(uniqueContries),1,i+1)
            if i == 0:
                plt.title(singleChannelKayakDataFrame.keys()[-1] + ' for ' + channel)
            thisKayakDataFrame = singleChannelKayakDataFrame[singleChannelKayakDataFrame.country_code == country]

            print 'working on ' + country + ' ' + channel

            #check if there is a low number samples
            if thisKayakDataFrame.shape[0] < 200:
                thisKayakDataFrame = replaceMissingDaysWithZero(thisKayakDataFrame)
                thisKayakDataFrame = replaceMissingDaysWithZero(thisKayakDataFrame)
                channelVisitsEstimate[channel].append(thisKayakDataFrame['user_visits'].mean())

                endMonthMedian = thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].loc[pd.date_range(dt.datetime(2015,8,01,00,00,00), periods=31, freq='D')].median()
                earlierMonthMedian = thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].loc[pd.date_range(dt.datetime(2015,4,01,00,00,00), periods=31, freq='D')].median()
                if endMonthMedian > 3.0*earlierMonthMedian:
                    #then assum trending
                    print ' trending'
                    appended = appendTimeProxi(thisKayakDataFrame)
                    plt.plot(appended[appended.keys()[-1]].values,'r')
                    plt.ylabel(country)
                else:
                    appended = appendTimeProxi(thisKayakDataFrame)
                    plt.plot(appended[appended.keys()[-1]].values,'k')
                    plt.ylabel(country)
                continue

            #replace missing data with epsilon values
            thisKayakDataFrame = replaceMissingDaysWithZero(thisKayakDataFrame)

            #look at power ratio of weekly occilations
            weekPowerInStds = plotFourrie(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].values,plots = 0)

            '''
            #handle cases differently when there is a low overall number
            if thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].median() < 100:
                print country + ' ' + channel + ' median of ' + str(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].median()) + ' std of ' + str(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].std())
                appended = appendTimeProxi(thisKayakDataFrame)
                plt.subplot(len(uniqueContries),1,i+1)
                plt.plot(appended[appended.keys()[-1]].values)
                plt.plot(np.array([1,appended.shape[0]]),appended[appended.keys()[-1]].mean()*np.ones(2),'k',linewidth = 2)
                plt.ylabel(country)
            '''

            if weekPowerInStds > 2.5:
                copyKayakDataFrame = thisKayakDataFrame.copy('deep')
                del copyKayakDataFrame['marketing_channel']
                del copyKayakDataFrame['country_code']
                trended = trendDecomposition(copyKayakDataFrame,channel2detrend='user_visits')
                appended = appendTimeProxi(trended)
                appended['trend'] = appended['trend'].interpolate()
                plt.subplot(len(uniqueContries),1,i+1)
                plt.plot(appended['trend'].values)
                plt.plot(appended[appended.keys()[0]].values)
                plt.ylabel(country)
                channelVisitsEstimate[channel].append(appended['trend'].loc[pd.date_range(dt.datetime(2015,9,01,00,00,00), periods=30, freq='D')].mean())
            else:
                thisKayakDataFrame['rollingMeanVisits'] = pd.rolling_mean(thisKayakDataFrame['user_visits'],14)
                #nanIdx = np.where(np.isnan(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].values))[0]
                #thisKayakDataFrameNoNans = thisKayakDataFrame.dropna(subset = ['rollingMeanRate'])

                appended = appendTimeProxi(thisKayakDataFrame)
                appended = appended.interpolate()
                plt.subplot(len(uniqueContries),1,i+1)
                plt.plot(appended['rollingMeanVisits'].values)
                plt.ylabel(country)

                channelVisitsEstimate[channel].append(appended['user_visits'].loc[pd.date_range(dt.datetime(2015,9,01,00,00,00), periods=30, freq='D')].mean())

        print channel
        if savepath:
            plt.savefig(savepath + str(channel) + '.jpg')
            plt.close()
    return channelVisitsEstimate


mergedFields = findRate()

plt.close('all')
#channelRates = plotRateEstimates(mergedFields)
estimatedSeptRates = plotRateEstimates(mergedFields,savepath = './ratePredictions_')
visits = pd.read_csv('E:/applications/KAYAK/data_files/visits.csv',index_col = 'datestamp',parse_dates=True)
estimatedSeptVisits = plotVisitsEstimates(visits,savepath = './vistsPredictions_')

#rollingMean = pd.rolling_mean(df['Close'], 100)

for marketikngChannel in estimatedSeptVisits.keys():
    theseVisits = np.array(estimatedSeptVisits[marketikngChannel])
    wAvg = np.average(estimatedSeptRates[marketikngChannel],weights = theseVisits/theseVisits.sum())
    print marketikngChannel + ' has a predicted average of ' + str(wAvg) ' for the month Sept.'

