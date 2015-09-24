import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


###
# Question 1
###

conversions = pd.read_csv('./data_files/conversions.csv',index_col = 'datestamp',parse_dates=True)
visits = pd.read_csv('./data_files/visits.csv',index_col = 'datestamp',parse_dates=True)

def plotByCountryAndMarket(kayakDataFrame,savepath = None):
    for country in set(kayakDataFrame.country_code):
        restrictedKayakDataFrame = kayakDataFrame[(kayakDataFrame.country_code == country)]
        fig = plt.figure(figsize=(15, 12))
        uniqueChannels = set(restrictedKayakDataFrame.marketing_channel)
        for i, channel in enumerate(uniqueChannels):
            singleChannelKayakDataFrame = restrictedKayakDataFrame[(restrictedKayakDataFrame.marketing_channel == channel)]
            plt.subplot(len(uniqueChannels),1,i+1)
            if i == 0:
                plt.title(singleChannelKayakDataFrame.keys()[-1] + ' for ' + country)
            plt.figure(fig.number)
            singleChannelKayakDataFrame[singleChannelKayakDataFrame.keys()[-1]].plot()
            plt.xlabel('time')
            plt.ylabel(channel)
        if savepath:
            plt.savefig(savepath + country + '.jpg')
            plt.close()

def plotByMarketAndCountry(kayakDataFrame,savepath = None):
    for channel in set(conversions.marketing_channel):
        restrictedKayakDataFrame = kayakDataFrame[(kayakDataFrame.marketing_channel == channel)]
        fig = plt.figure(figsize=(15, 12))
        uniqueContries = set(restrictedKayakDataFrame.country_code)
        for i, country in enumerate(uniqueContries):
            singleCountryKayakDataFrame = restrictedKayakDataFrame[(restrictedKayakDataFrame.country_code == country)]
            plt.subplot(len(uniqueContries),1,i+1)
            if i == 0:
                plt.title(singleCountryKayakDataFrame.keys()[-1] + ' for ' + channel)
            singleCountryKayakDataFrame[singleCountryKayakDataFrame.keys()[-1]].plot()
            plt.xlabel('time')
            plt.ylabel(country)
        if savepath:
            plt.savefig(savepath + channel + '.jpg')
            plt.close()

def plotFourrie(npArray):
    samples2use = int(2**np.floor(np.log2(npArray.shape[0])))
    fftVals = np.fft.fft(npArray.astype(int),n=samples2use,axis=-1)
    #fftVals = np.fft.fft(npArray.astype(int))
    power = np.abs(fftVals)**2
    freqs = np.fft.fftfreq(samples2use, 1)
    plt.figure()
    #plt.plot(power)


    plt.plot(np.fft.fftshift(freqs),np.fft.fftshift(power))

    freqs = np.fft.fftshift(freqs)
    power = np.fft.fftshift(power)
    weekpeak = power[np.where(freqs < 1.0/6.0) and np.where(freqs > 1.0/8.0)].max()
    globalpeak= power[np.where(freqs < 1.0/2.5) and np.where(freqs > 1.0/30)].max()
    return weekpeak/globalpeak


plotByCountryAndMarket(conversions,'./conversions_')
plotByCountryAndMarket(visits,'./visits_')

plotByMarketAndCountry(conversions,'./conversions_')
plotByMarketAndCountry(visits,'./visits_')

conversionValues = conversions.conversions.values
grouptedRateMed = pd.rolling_median(grouptedRate,14)

###
# Question 2
###

conversions.reset_index(level=0, inplace=True)
visits.reset_index(level=0, inplace=True)
visitsAndConversions = pd.merge(visits, conversions, how='outer', on=['datestamp', 'country_code','marketing_channel'])
visitsAndConversions = visitsAndConversions.set_index('datestamp')
visitsAndConversions['rate'] = visitsAndConversions['conversions']/visitsAndConversions['user_visits']
#rate will be the last key, now we can use our origional functions to plot across country and market
plotByMarketAndCountry(visitsAndConversions,savepath = './rates_')

grouptedRate = visitsAndConversions.groupby(visitsAndConversions.index).sum()
grouptedRate['rate'] = grouptedRate['conversions']/grouptedRate['user_visits']
grouptedRate['rate'].plot()
