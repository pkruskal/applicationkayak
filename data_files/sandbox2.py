import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
channel_descriptions = channelDescritions = pd.read_csv('./data_files/channel_descriptions.csv')

'''
channel_descriptions.keys()
'marketing_channel', u'example_user_acquisition_site', u'description'
'''

# (9006, 4)
# datestamp country_code      marketing_channel  conversions
conversions = pd.read_csv('./data_files/conversions.csv',index_col = 'datestamp',parse_dates=True)

fig = plt.figure()
conversions.conversions.plot()
plt.title('all conversions over time')
plt.xlabel('time')
plt.ylabel('conversions')


def plotConverstionByCountryAndMarket(conversions):
    for country in set(conversions.country_code):
        theseConversions = conversions[(conversions.country_code == country)]
        fig = plt.figure()
        uniqueChannels = set(theseConversions.marketing_channel)
        for i, channel in enumerate(uniqueChannels):
            plt.subplot(len(uniqueChannels),1,i+1)
            if i == 0:
                plt.title('conversions for ' + country)
            theseConversions[(theseConversions.marketing_channel == channel)].conversions.plot()
            plt.xlabel('time')
            plt.ylabel(channel)


def plotConverstionByMarketAndCountry(conversions):
    for channel in set(conversions.marketing_channel):
        theseConversions = conversions[(conversions.marketing_channel == channel)]
        fig = plt.figure()
        uniqueContries = set(theseConversions.country_code)
        for i, country in enumerate(uniqueContries):
            plt.subplot(len(uniqueContries),1,i+1)
            if i == 0:
                plt.title('conversions for ' + channel)
            theseConversions[(theseConversions.country_code == country)].conversions.plot()
            plt.xlabel('time')
            plt.ylabel(country)


plt.close('all')

fig = plt.figure()
theseConversions[(theseConversions.marketing_channel == channel)].conversions.plot('hist')
theseConversions[(theseConversions.marketing_channel == channel)].conversions.plot('kde')


visits = pd.read_csv('./data_files/visits.csv',index_col = 'datestamp',parse_dates=True)


def fourrieAnalysis(npArray):
    samples2use = 2**np.floor(np.log2(npArray.shape[0]))
    np.fft.fft(npArray,n=samples2use)


####
# objective is to evaluate conversion rate
# can mean different things
# flight, hotel booking, click on a specific part of the site, an account sign up

def ploVisitsByMarketAndCountry(visits):
    for channel in set(visits.marketing_channel):
        theseVisits = visits[(visits.marketing_channel == channel)]
        fig = plt.figure()
        uniqueContries = set(theseVisits.country_code)
        for i, country in enumerate(uniqueContries):
            plt.subplot(len(uniqueContries),1,i+1)
            if i == 0:
                plt.title('visits for ' + channel)
            theseVisits[(theseVisits.country_code == country)].user_visits.plot()
            plt.xlabel('time')
            plt.ylabel(country)


def seasonalDecompositionByStatsModelExample():
    import statsmodels.api as sm

    res = sm.tsa.seasonal_decompose(groupedOverCountry.user_visits.values,model="multiplicative",freq=7)

    resplot = res.plot()

def plotConversionReteByChannel():
    plt.figure()
    for channel in set(conversions.marketing_channel):
        theseConversions = conversions[(conversions.marketing_channel == channel)]
        theseVisits = visits[(visits.marketing_channel == channel)]

        #index on country and date to align for the join command to add visits
        theseConversions.reset_index(level=0, inplace=True)
        theseVisits.reset_index(level=0, inplace=True)
        valsAndConversions = pd.merge(theseVisits, theseConversions, how='outer', on=['datestamp', 'country_code','marketing_channel'])
        valsAndConversions = valsAndConversions.set_index('datestamp')

        #for this question we wont consider countries sepperatly
        #group over countries by data (which is the index)
        groupedOverCountry = valsAndConversions.groupby(valsAndConversions.index).sum()

        #get the rate
        groupedOverCountry['rate'] = groupedOverCountry['conversions']/groupedOverCountry['user_visits']

        groupedOverCountry['rate'].plot()
    plt.legend(set(conversions.marketing_channel))

def seasonalTrend(thisSeries,days):
    weekList = []
    weekAmpList = []
    weekMeanList = []
    plt.figure()
    for iWeek in set(groupedOverCountry.index.week):
        thisWeek = np.array(groupedOverCountry[(groupedOverCountry.index.week == iWeek)].user_visits.values)
        if thisWeek.shape[0] == 7:
            demeanedWeek = thisWeek-np.mean(thisWeek)
            weekAmp = np.max(thisWeek)-np.min(thisWeek)
            weekAmpList.append(weekAmp)
            weekMeanList.append(np.mean(thisWeek))
            weekList.append(thisWeek)
            plt.subplot(3,1,1)
            plt.plot(demeanedWeek/weekAmp)
            plt.subplot(3,1,2)
            plt.plot(demeanedWeek)
            plt.subplot(3,1,3)
            plt.plot(thisWeek)
    return weekList, weekAmpList, weekMeanList

plt.plot(weekMeanList,weekAmpList,'.')
#via observation
# use 45000 as a cut off for seasonal adjustment
# use a linear fit of seasonal adjustment

weekMeanArray = np.array(weekMeanList)
weekArray = np.array(weekList)
np.where(weekMeanArray>45000)

#note day 0 is monday





import datetime as dt
visits = pd.read_csv('./data_files/visits.csv',index_col = 'datestamp',parse_dates=True)
visits = visits.groupby(visits.index).sum()

#adding as missing data

#extending Nov and Dec as from year 2014
base = dt.datetime(2015,9,01,00,00,00)
septToOct = [base + dt.timedelta(days=x) for x in range(0, 61)]
for date in septToOct:
    visits.loc[date] = np.nan

for day in np.arange(1, 62):
    visits.loc[septToOct[-1] + dt.timedelta(days=day)] = visits.user_visits[day-1]

visitsBasic = visits.interpolate()
visitsCupic = visits.interpolate(method = 'cubic')
visitsQuadratic = visits.interpolate(method = 'quadratic')
visitsQuadratic = visits.interpolate(method = 'spline',order = 5)