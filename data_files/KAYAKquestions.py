import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime as dt


########
# helper functions
########


def removeOutlier4Conversions(conversions):
    '''
    Drop extream values (greater than 100000)
    :param conversions dataframe:
    :return: outlier removed data frame
    '''
    conversions = conversions.loc[conversions['conversions'] < 100000]
    return conversions

def fourierAnalysis(npArray,plots = 0):
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

def trendDecomposition(kayakDF2detrend,channel2detrend = None):
    '''
    uses pythons sm.tsa.seasonal_decompose module function to isolate out weekly fluctuations in a trace
    :param kayakDF2detrend:
    :param channel2detrend:
    :return: detrended trace
    '''

    #if no pandas column is selected to detrent, use the last column
    if channel2detrend == None:
        channel2detrend = kayakDF2detrend.keys()[-1]

    #for numerical reasons the we can have no 0's for the multiplicative detrending model
    zeroidx = np.where(kayakDF2detrend[channel2detrend].values == 0)[0]
    kayakDF2detrend[channel2detrend][zeroidx] = np.finfo(float).eps

    #weekly decomposition (freq = 7) with a multiplicitave model
    res = sm.tsa.seasonal_decompose(kayakDF2detrend[channel2detrend].values,model="multiplicative",freq=7)
    kayakDF2detrend['trend'] = res.trend
    kayakDF2detrend['seasonal'] = res.seasonal+res.trend
    return kayakDF2detrend

def plotByCountryAndMarket(kayakDataFrame,savepath = None):
    '''
    Creates a seprate figure for every country and a subplot in the figure for every market channel

    For each channel the function looks at the power in a weekly frequency and, if high enough,
    performs a decomposition of the trace to isolate weekly oscillations

    :param kayakDataFrame: loaded data frame from kayak to plot
    :param savepath: where to save figures
    '''

    #first loop through all the countries in the data frame
    for country in set(kayakDataFrame.country_code):

        #query just entires on the country
        restrictedKayakDataFrame = kayakDataFrame[(kayakDataFrame.country_code == country)]

        #build the figure
        fig = plt.figure(figsize=(15, 12))

        #loop through all the marketing channels for the cnountry
        uniqueChannels = set(restrictedKayakDataFrame.marketing_channel)
        for i, channel in enumerate(uniqueChannels):

            #queyr just the entries with the maketing channel
            singleChannelKayakDataFrame = restrictedKayakDataFrame[(restrictedKayakDataFrame.marketing_channel == channel)]

            #build the subplot
            plt.subplot(len(uniqueChannels),1,i+1)
            #if it's the first subplot title the figure
            if i == 0:
                plt.title(singleChannelKayakDataFrame.keys()[-1] + ' for ' + country)

            #check the relative strengh of weekly power by looking at the number
            #of standard deviations weekly power is above the near by power distribution
            weeelyPowSTDs = fourierAnalysis(singleChannelKayakDataFrame[singleChannelKayakDataFrame.keys()[-1]].values,plots = 0)

            #if weekly power is over 2.5 standard deviations above the mean then isolate weekly fluctuations
            if weeelyPowSTDs > 2.5:
                #some formating for trend analysis
                copyKayakDataFrame = singleChannelKayakDataFrame.copy('deep')
                del copyKayakDataFrame['marketing_channel']
                del copyKayakDataFrame['country_code']

                #isolate the weekly component useing pythons statsmodels
                trended = trendDecomposition(copyKayakDataFrame)

                #refresh the subplot (fixes a weird bug in matplot lib)
                plt.subplot(len(uniqueChannels),1,i+1)

                #plot the trace components
                plt.plot(trended['trend'].values)
                plt.plot(trended[trended.keys()[0]].values)
                plt.plot(trended['seasonal'].values,'y')
                plt.xlabel('time')
                plt.ylabel(channel)
            else:
                #refresh the subplot (fixes a weird bug in matplot lib)
                plt.subplot(len(uniqueChannels),1,i+1)

                #plot the trace
                singleChannelKayakDataFrame[singleChannelKayakDataFrame.keys()[-1]].plot()
                plt.xlabel('time')
                plt.ylabel(channel)

        #if a save path is specified, save the figure and close it
        if savepath:
            plt.savefig(savepath + country + '.jpg')
            plt.close()

def plotByMarketAndCountry(kayakDataFrame,savepath = None):
    '''
    Similar to plot by country and market, but creates a seperate plot for each market, and a subplot for each country
    This function does not bother with taking out weekly fluctuations
    :param kayakDataFrame:
    :param savepath:
    :return:
    '''
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

def findRate():
    '''
    performs a join on the visits and converstion channels to align for the rate computation
    :return:
    '''
    #conversions = pd.read_csv('D:/KAYAKtest/data_files/conversions.csv',index_col = 'datestamp',parse_dates=True)
    #visits = pd.read_csv('D:/KAYAKtest/data_files/visits.csv',index_col = 'datestamp',parse_dates=True)
    conversions = pd.read_csv('E:/applications/KAYAK/data_files/conversions.csv',index_col = 'datestamp',parse_dates=True)
    visits = pd.read_csv('E:/applications/KAYAK/data_files/visits.csv',index_col = 'datestamp',parse_dates=True)

    #remove outlier
    conversions = removeOutlier4Conversions(conversions)

    #perform join them
    conversions.reset_index(level=0, inplace=True)
    visits.reset_index(level=0, inplace=True)
    visitsAndConversions = pd.merge(visits, conversions, how='outer', on=['datestamp', 'country_code','marketing_channel'])
    visitsAndConversions = visitsAndConversions.set_index('datestamp')

    #if there is a visit value, but no conversion value (ie nan) then set conversion to 0
    visitsAndConversions.conversions.values[[[np.array(~np.isnan(visitsAndConversions.user_visits)) * np.array(np.isnan(visitsAndConversions.conversions))][0]]] = 0

    #compute rate
    visitsAndConversions['rate'] = visitsAndConversions['conversions']/visitsAndConversions['user_visits']
    return visitsAndConversions

def appendTimeProxi(dFrame2append):
    '''
    This function extends the data frame to Dec 2015
    It uses NaNs fro Sept. and Oct
    It uses 2014s data for Nov. and Dec.
    :param the KAYAK data frame to extend in time
    :return: extended data frame
    '''

    #extend for Sept. and Oct.
    base = dt.datetime(2015,9,01,00,00,00)
    septToOct = [base + dt.timedelta(days=x) for x in range(0, 61)]
    for date in septToOct:
        dFrame2append.loc[date] = np.nan

    #taggin on nov and dec 2014 as proxi for 2015
    for day in np.arange(1, 62):
        dFrame2append.loc[septToOct[-1] + dt.timedelta(days=day)] = dFrame2append.loc[dt.datetime(2014,11,01,00,00,00) + dt.timedelta(days=day-1)]

    return dFrame2append

def replaceMissingDaysWithZero(kayakDataFrameWithMissing):
    '''
    Fills in all day in the datetime index.
    If a day does not exist it's replaced with an eps value

    like many functions used here, I take advantage of the fact that in these
    data frames only the last key has the numberical value of interest
    I use this to generalize my function for conversions and visits
    '''

    key2Edit = kayakDataFrameWithMissing.keys()[-1]

    #make sure there is a time index for a consistent range
    idx = pd.date_range(dt.datetime(2014,11,01,00,00,00),dt.datetime(2015,8,31,00,00,00),freq = 'D')
    kayakRedex = kayakDataFrameWithMissing.reindex(idx)

    #the assumption here is that the data is missing because it is 0 so replace Nan values with epsilon
    #epsilon is used here for numerical stability in later calculations
    kayakRedex[key2Edit][np.where(np.isnan(kayakRedex[key2Edit]))[0]] = np.finfo(float).eps

    return kayakRedex

def plotRateEstimates(kayakDataFrame,savepath=None):
    '''
    Loops over marketing channel and aproximates the Sept. convertion rates for each channel and country in a dictionary
    The approximation is achieved by smoothing and then interpolating
    between the month of Aug. and the month of Nov on the previous year
    If less then 200 events occur over the 10 months given, then only the mean conversion rate is used as an estimate
    for Nov. converstions
    All the analysis is ploted for visual inspection
    :param kayakDataFrame:
    :param savepath:
    :return: channelRates: a dictionary where every key is a marketing channel,
    each marketing channel provides a list of all countries rate estimates
    '''

    #initalize dictionary to store all results
    channelRates = {}

    for channel in set(kayakDataFrame.marketing_channel):

        #add to dictionary
        channelRates[channel] = []

        #query marketing channel
        singleChannelKayakDataFrame = kayakDataFrame[(kayakDataFrame.marketing_channel == channel)]

        #initalize figure
        fig = plt.figure(figsize=(15, 12))

        #now estimate each country independently
        uniqueContries = set(singleChannelKayakDataFrame.country_code)
        for i, country in enumerate(uniqueContries):

            plt.subplot(len(uniqueContries),1,i+1)
            #add title if it's the first subplot
            if i == 0:
                plt.title(singleChannelKayakDataFrame.keys()[-1] + ' for ' + channel)

            #querry country
            thisKayakDataFrame = singleChannelKayakDataFrame[singleChannelKayakDataFrame.country_code == country]

            #check if there is a low number samples
            if thisKayakDataFrame.shape[0] < 200:
                #to get an accurate mean, all days must be accounted for with 0 (eps) values if they don't exist
                thisKayakDataFrame = replaceMissingDaysWithZero(thisKayakDataFrame)

                #add the mean rate and store it
                channelRates[channel].append(thisKayakDataFrame['rate'].mean())

                #append the time for consistencies sake in the plot
                appended = appendTimeProxi(thisKayakDataFrame)

                #plot the results
                plt.plot(appended[appended.keys()[-1]].values,'k')
                plt.ylabel(country)

                #move on to the next country
                continue

            #replace missing data with epsilon values
            thisKayakDataFrame = replaceMissingDaysWithZero(thisKayakDataFrame)

            #smooth out the trace with a box car filter
            thisKayakDataFrame['rollingMeanRate'] = pd.rolling_mean(thisKayakDataFrame['rate'],14)

            #append on Nov and Dec 2014 as proxies for Nov and Dec 2015
            appended = appendTimeProxi(thisKayakDataFrame)

            #perform linear interpolation
            appended = appended.interpolate()

            #plot the results
            plt.subplot(len(uniqueContries),1,i+1)
            plt.plot(appended['rollingMeanRate'].values)
            plt.ylabel(country)

            #append the mean interpolated estimate for Sept.
            channelRates[channel].append(appended['rollingMeanRate'].
                                         loc[pd.date_range(dt.datetime(2015,9,01,00,00,00), periods=30, freq='D')].mean())
        if savepath:
            #save an close the figure
            plt.savefig(savepath + str(channel) + '.jpg')
            plt.close()
    return channelRates

def plotVisitsEstimates(kayakDataFrame,savepath=None):
    '''
    similar to plotRateEstimates but for visits
    Also checks and removes weekly occilations
    :param kayakDataFrame:
    :param savepath:
    :return: channelVisitsEstimate
    '''

    #store all the estimates by channel in this dictionary
    channelVisitsEstimate = {}

    #loop through channels
    for channel in set(kayakDataFrame.marketing_channel):

        #initalize this channels estimate list
        channelVisitsEstimate[channel] = []

        #querry channel
        singleChannelKayakDataFrame = kayakDataFrame[(kayakDataFrame.marketing_channel == channel)]

        #initalize figure
        fig = plt.figure(figsize=(15, 12))

        #loop through countries treating each independently
        uniqueContries = set(singleChannelKayakDataFrame.country_code)
        for i, country in enumerate(uniqueContries):


            plt.subplot(len(uniqueContries),1,i+1)
            if i == 0:
                plt.title(singleChannelKayakDataFrame.keys()[-1] + ' for ' + channel)

            #querry country
            thisKayakDataFrame = singleChannelKayakDataFrame[singleChannelKayakDataFrame.country_code == country]

            #check if there is a low number samples
            if thisKayakDataFrame.shape[0] < 200:

                #add non entries as epsilon
                thisKayakDataFrame = replaceMissingDaysWithZero(thisKayakDataFrame)

                #store the mean as the visits aproximation
                channelVisitsEstimate[channel].append(thisKayakDataFrame['user_visits'].mean())

                #plot
                appendTimeProxi(thisKayakDataFrame)
                plt.plot(appended[appended.keys()[-1]].values,'k')
                plt.ylabel(country)
            continue

            #replace missing data with epsilon values
            thisKayakDataFrame = replaceMissingDaysWithZero(thisKayakDataFrame)

            #look at power of weekly oscillations as in question 1 plots
            weekPowerInStds = plotFourrie(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].values,plots = 0)

            if weekPowerInStds > 2.5:

                #format the data
                copyKayakDataFrame = thisKayakDataFrame.copy('deep')
                del copyKayakDataFrame['marketing_channel']
                del copyKayakDataFrame['country_code']

                #decompose the data
                trended = trendDecomposition(copyKayakDataFrame,channel2detrend='user_visits')

                #add on Nov and Dec 2014 as proxies for 2015
                appended = appendTimeProxi(trended)

                #linear interpolation
                appended['trend'] = appended['trend'].interpolate()

                #store the results
                channelVisitsEstimate[channel].append(appended['trend'].loc[pd.date_range(dt.datetime(2015,9,01,00,00,00), periods=30, freq='D')].mean())

                #plot the results
                plt.subplot(len(uniqueContries),1,i+1)
                plt.plot(appended['trend'].values)
                plt.plot(appended[appended.keys()[0]].values)
                plt.ylabel(country)


            else:
                #smooth the data wit ha box car filter
                thisKayakDataFrame['rollingMeanVisits'] = pd.rolling_mean(thisKayakDataFrame['user_visits'],14)
                #nanIdx = np.where(np.isnan(thisKayakDataFrame[thisKayakDataFrame.keys()[-1]].values))[0]
                #thisKayakDataFrameNoNans = thisKayakDataFrame.dropna(subset = ['rollingMeanRate'])

                #add on Nov and Dec 2014 as proxies for 2015
                appended = appendTimeProxi(thisKayakDataFrame)

                #linear interpolation
                appended = appended.interpolate()

                #store the results
                channelVisitsEstimate[channel].append(appended['user_visits'].loc[pd.date_range(dt.datetime(2015,9,01,00,00,00), periods=30, freq='D')].mean())

                #plot the results
                plt.subplot(len(uniqueContries),1,i+1)
                plt.plot(appended['rollingMeanVisits'].values)
                plt.ylabel(country)

        if savepath:
            plt.savefig(savepath + str(channel) + '.jpg')
            plt.close()
    return channelVisitsEstimate


##########
# Question 1
##########

#load the data
conversions = pd.read_csv('./data_files/conversions.csv',index_col = 'datestamp',parse_dates=True)
visits = pd.read_csv('./data_files/visits.csv',index_col = 'datestamp',parse_dates=True)

#remove extream outliers from the data
conversions = removeOutlier4Conversions(conversions)

#plot all channels by country and market for both visits and conversions
plotByCountryAndMarket(conversions,'./conversions_')
plotByCountryAndMarket(visits,'./visits_')


###########
# Question 2
###########

mergedFields = findRate

#Now we can use our original functions to plot across country and market
plotByMarketAndCountry(mergedFields[['marketing_channel','country_code','rate']],savepath = './rates_')

#group the rates to summarize the comparison for marketing channel
grouptedRateMedian = mergedFields.groupby(mergedFields.marketing_channel).median()
grouptedRateMean = mergedFields.groupby(mergedFields.marketing_channel).mean()
grouptedRateStd = mergedFields.groupby(mergedFields.marketing_channel).std()
groupRatedCount = mergedFields.groupby(mergedFields.marketing_channel).count()
grouptedRateMean['ste'] = grouptedRateStd['rate']/np.sqrt(groupRatedCount['rate'])


###########
# Question 4
###########


estimatedSeptRates = plotRateEstimates(mergedFields,savepath = './ratePredictions_')
visits = pd.read_csv('E:/applications/KAYAK/data_files/visits.csv',index_col = 'datestamp',parse_dates=True)
estimatedSeptVisits = plotVisitsEstimates(visits,savepath = './vistsPredictions_')

#rollingMean = pd.rolling_mean(df['Close'], 100)

for marketikngChannel in estimatedSeptVisits.keys():
    theseVisits = np.array(estimatedSeptVisits[marketikngChannel])
    wAvg = np.average(estimatedSeptRates[marketikngChannel],weights = theseVisits/theseVisits.sum())
    print marketikngChannel + ' has a predicted average of ' + str(wAvg) + ' for the month Sept.'


