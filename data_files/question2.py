
def findRate():
    conversions = pd.read_csv('D:/KAYAKtest/data_files/conversions.csv',index_col = 'datestamp',parse_dates=True)
    visits = pd.read_csv('D:/KAYAKtest/data_files/visits.csv',index_col = 'datestamp',parse_dates=True)

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


#group by marketing channel
grouptedRateMedian = visitsAndConversions.groupby(visitsAndConversions.marketing_channel).median()
grouptedRateMean = visitsAndConversions.groupby(visitsAndConversions.marketing_channel).mean()
#grouptedRateMad = visitsAndConversions.groupby(visitsAndConversions.marketing_channel).mad()
grouptedRateStd = visitsAndConversions.groupby(visitsAndConversions.marketing_channel).std()
groupRatedCount = visitsAndConversions.groupby(visitsAndConversions.marketing_channel).count()
grouptedRateMean['ste'] = grouptedRateStd['rate']/np.sqrt(groupRatedCount['rate'])

