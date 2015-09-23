import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

conversions = pd.read_csv('./data_files/conversions.csv',index_col = 'datestamp',parse_dates=True)
visits = pd.read_csv('./data_files/visits.csv',index_col = 'datestamp',parse_dates=True)

def plotByCountryAndMarket(kayakDataFrame,savepath = None):
    for country in set(kayakDataFrame.country_code):
        print country
        restrictedKayakDataFrame = kayakDataFrame[(kayakDataFrame.country_code == country)]
        fig = plt.figure()
        uniqueChannels = set(restrictedKayakDataFrame.marketing_channel)
        for i, channel in enumerate(uniqueChannels):
            plt.subplot(len(uniqueChannels),1,i+1)
            if i == 0:
                plt.title('conversions for ' + country)
            restrictedKayakDataFrame[(restrictedKayakDataFrame.marketing_channel == channel)].conversions.plot()
            plt.xlabel('time')
            plt.ylabel(channel)
        if savepath:
            plt.savefig(savepath + country + '.jpg')
            plt.close()

plotByCountryAndMarket(conversions,'./conversions')
plotByCountryAndMarket(visits,'./visits')

conversionValues = conversions.conversions.values
def plotFourrie(npArray):
    samples2use = int(2**np.floor(np.log2(npArray.shape[0])))
    fftVals = np.fft.fft(npArray.astype(int),n=samples2use,axis=-1)
    #fftVals = np.fft.fft(npArray.astype(int))
    power = np.abs(fftVals)**2
    freqs = np.fft.fftfreq(samples2use, 1)
    plt.figure()
    plt.plot(power)


    plt.plot(np.fft.fftshift(freqs),np.fft.fftshift(power))

    freqs = np.fft.fftshift(freqs)
    power = np.fft.fftshift(power)
    weekpeak = power[np.where(freqs < 1.0/6.0) and np.where(freqs > 1.0/8.0)].max()
    globalpeak= power[np.where(freqs < 1.0/2.5) and np.where(freqs > 1.0/30)].max()