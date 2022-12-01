from pyaxis import pyaxis
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
import missingno as msno

datelist = [pd.Timestamp('2013-01-01 00:00:00'),
 pd.Timestamp('2013-02-01 00:00:00'),
 pd.Timestamp('2013-03-01 00:00:00'),
 pd.Timestamp('2013-04-01 00:00:00'),
 pd.Timestamp('2013-05-01 00:00:00'),
 pd.Timestamp('2013-06-01 00:00:00'),
 pd.Timestamp('2013-07-01 00:00:00'),
 pd.Timestamp('2013-08-01 00:00:00'),
 pd.Timestamp('2013-09-01 00:00:00'),
 pd.Timestamp('2013-10-01 00:00:00'),
 pd.Timestamp('2013-11-01 00:00:00'),
 pd.Timestamp('2013-12-01 00:00:00'),
 pd.Timestamp('2014-01-01 00:00:00'),
 pd.Timestamp('2014-02-01 00:00:00'),
 pd.Timestamp('2014-03-01 00:00:00'),
 pd.Timestamp('2014-04-01 00:00:00'),
 pd.Timestamp('2014-05-01 00:00:00'),
 pd.Timestamp('2014-06-01 00:00:00'),
 pd.Timestamp('2014-07-01 00:00:00'),
 pd.Timestamp('2014-08-01 00:00:00'),
 pd.Timestamp('2014-09-01 00:00:00'),
 pd.Timestamp('2014-10-01 00:00:00'),
 pd.Timestamp('2014-11-01 00:00:00'),
 pd.Timestamp('2014-12-01 00:00:00'),
 pd.Timestamp('2015-01-01 00:00:00'),
 pd.Timestamp('2015-02-01 00:00:00'),
 pd.Timestamp('2015-03-01 00:00:00'),
 pd.Timestamp('2015-04-01 00:00:00'),
 pd.Timestamp('2015-05-01 00:00:00'),
 pd.Timestamp('2015-06-01 00:00:00'),
 pd.Timestamp('2015-07-01 00:00:00'),
 pd.Timestamp('2015-08-01 00:00:00'),
 pd.Timestamp('2015-09-01 00:00:00'),
 pd.Timestamp('2015-10-01 00:00:00'),
 pd.Timestamp('2015-11-01 00:00:00'),
 pd.Timestamp('2015-12-01 00:00:00'),
 pd.Timestamp('2016-01-01 00:00:00'),
 pd.Timestamp('2016-02-01 00:00:00'),
 pd.Timestamp('2016-03-01 00:00:00'),
 pd.Timestamp('2016-04-01 00:00:00'),
 pd.Timestamp('2016-05-01 00:00:00'),
 pd.Timestamp('2016-06-01 00:00:00'),
 pd.Timestamp('2016-07-01 00:00:00'),
 pd.Timestamp('2016-08-01 00:00:00'),
 pd.Timestamp('2016-09-01 00:00:00'),
 pd.Timestamp('2016-10-01 00:00:00'),
 pd.Timestamp('2016-11-01 00:00:00'),
 pd.Timestamp('2016-12-01 00:00:00'),
 pd.Timestamp('2017-01-01 00:00:00'),
 pd.Timestamp('2017-02-01 00:00:00'),
 pd.Timestamp('2017-03-01 00:00:00'),
 pd.Timestamp('2017-04-01 00:00:00'),
 pd.Timestamp('2017-05-01 00:00:00'),
 pd.Timestamp('2017-06-01 00:00:00'),
 pd.Timestamp('2017-07-01 00:00:00'),
 pd.Timestamp('2017-08-01 00:00:00'),
 pd.Timestamp('2017-09-01 00:00:00'),
 pd.Timestamp('2017-10-01 00:00:00'),
 pd.Timestamp('2017-11-01 00:00:00'),
 pd.Timestamp('2017-12-01 00:00:00'),
 pd.Timestamp('2018-01-01 00:00:00'),
 pd.Timestamp('2018-02-01 00:00:00'),
 pd.Timestamp('2018-03-01 00:00:00'),
 pd.Timestamp('2018-04-01 00:00:00'),
 pd.Timestamp('2018-05-01 00:00:00'),
 pd.Timestamp('2018-06-01 00:00:00'),
 pd.Timestamp('2018-07-01 00:00:00'),
 pd.Timestamp('2018-08-01 00:00:00'),
 pd.Timestamp('2018-09-01 00:00:00'),
 pd.Timestamp('2018-10-01 00:00:00'),
 pd.Timestamp('2018-11-01 00:00:00'),
 pd.Timestamp('2018-12-01 00:00:00'),
 pd.Timestamp('2019-01-01 00:00:00'),
 pd.Timestamp('2019-02-01 00:00:00'),
 pd.Timestamp('2019-03-01 00:00:00'),
 pd.Timestamp('2019-04-01 00:00:00'),
 pd.Timestamp('2019-05-01 00:00:00'),
 pd.Timestamp('2019-06-01 00:00:00'),
 pd.Timestamp('2019-07-01 00:00:00'),
 pd.Timestamp('2019-08-01 00:00:00'),
 pd.Timestamp('2019-09-01 00:00:00'),
 pd.Timestamp('2019-10-01 00:00:00'),
 pd.Timestamp('2019-11-01 00:00:00'),
 pd.Timestamp('2019-12-01 00:00:00'),
 pd.Timestamp('2020-01-01 00:00:00'),
 pd.Timestamp('2020-02-01 00:00:00'),
 pd.Timestamp('2020-03-01 00:00:00'),
 pd.Timestamp('2020-04-01 00:00:00'),
 pd.Timestamp('2020-05-01 00:00:00'),
 pd.Timestamp('2020-06-01 00:00:00'),
 pd.Timestamp('2020-07-01 00:00:00'),
 pd.Timestamp('2020-08-01 00:00:00'),
 pd.Timestamp('2020-09-01 00:00:00'),
 pd.Timestamp('2020-10-01 00:00:00'),
 pd.Timestamp('2020-11-01 00:00:00'),
 pd.Timestamp('2020-12-01 00:00:00'),
 pd.Timestamp('2021-01-01 00:00:00'),
 pd.Timestamp('2021-02-01 00:00:00'),
 pd.Timestamp('2021-03-01 00:00:00'),
 pd.Timestamp('2021-04-01 00:00:00'),
 pd.Timestamp('2021-05-01 00:00:00'),
 pd.Timestamp('2021-06-01 00:00:00'),
 pd.Timestamp('2021-07-01 00:00:00'),
 pd.Timestamp('2021-08-01 00:00:00'),
 pd.Timestamp('2021-09-01 00:00:00'),
 pd.Timestamp('2021-10-01 00:00:00'),
 pd.Timestamp('2021-11-01 00:00:00'),
 pd.Timestamp('2021-12-01 00:00:00'),
 pd.Timestamp('2022-01-01 00:00:00'),
 pd.Timestamp('2022-02-01 00:00:00'),
 pd.Timestamp('2022-03-01 00:00:00'),
 pd.Timestamp('2022-04-01 00:00:00'),
 pd.Timestamp('2022-05-01 00:00:00'),
 pd.Timestamp('2022-06-01 00:00:00'),
 pd.Timestamp('2022-07-01 00:00:00'),
 pd.Timestamp('2022-08-01 00:00:00'),
 pd.Timestamp('2022-09-01 00:00:00')]

def dict_clean (px_data_url, regions):
    px = rf'{px_data_url}'
    px = pyaxis.parse(uri = px , encoding = 'ISO-8859-2')
    df = px['DATA']
    dict_meta = px['METADATA']
    bad_countries = ['Übriges Südamerika',
                     'Übriges Zentralamerika, Karibik',
                     'Übriges Nordafrika',
                     'Übriges Afrika',
                     'Übriges Westasien',
                     'Übriges Süd- und Ostasien',
                     'Übriges Europa']
    for m in range(0, len(bad_countries)):
        df = df.loc[df['Herkunftsland'] != bad_countries[m]]
    df = df.loc[(df['Indikator'] == 'Logiernächte') & (df['DATA'] != '"..."') & (df['DATA'] != '"......"')]
    df["Jahr"] = df["Jahr"].astype(int)
    df["DATA"] = df["DATA"].astype(int)
    df_months = df.loc[(df['Monat'] != 'Jahrestotal')]
    df_months = df_months.groupby('Gemeinde')
    all_regions = df_months['Gemeinde'].unique()
    dict_regions = [{} for k in range(0, len(regions))]
    for i in range(0, len(regions)):
        dict_regions[i][regions[i]] = df_months.get_group(regions[i])
        dict_regions[i][regions[i]].reset_index(inplace = True)
        dict_regions[i][regions[i]].rename(columns = {'Jahr' : 'year', 'Monat': 'month'}, inplace = True)
    trans_dict = {
        'Schweiz' : 'Switzerland',
        'Deutschland' : 'Germany',
        'Italien': 'Italy',
        'Frankreich' : 'France',
        'Österreich' : 'Austria',
        'Vereinigtes Königreich': 'United Kingdom',
        'Irland' : 'Ireland',
        'Niederlande' : 'Netherlands',
        'Belgien' : 'Belgium',
        'Luxemburg' : 'Luxembourg',
        'Dänemark' : 'Denmark',
        'Schweden' : 'Sweden',
        'Norwegen' : 'Norway',
        'Finnland' : 'Finland',
        'Spanien' : 'Spain',
        'Griechenland' : 'Greece',
        'Türkei' : 'Turkey',
        'Island' : 'Iceland',
        'Polen' : 'Poland',
        'Ungarn' : 'Hungary',
        'Bulgarien': 'Bulgaria',
        'Zypern' : 'Cyprus',
        'Vereinigte Staaten' : 'United States',
        'Kanada' : 'Canada',
        'Mexiko' : 'Mexico',
        'Brasilien' : 'Brasil',
        'Portugal' : 'Portugal',
        'Argentinien' : 'Argentina',
        'Ägypten' : 'Egypt',
        'Südafrika' : 'South Africa',
        'Indien' : 'India',
        'Katar' : 'Qatar',
        'Asutralien' : 'Australia',
        'Indonesien' : 'Indonesia',
        'Korea (Süd-)' : 'South Korea',
        'Phillipinen' : 'Philippines',
        'Neuseeland, Ozeanien' : 'New Zealand',
        'Singapur' : 'Singapore',
        'Taiwan (Chinesisches Taipei)' : 'Taiwan',
        'Estland' : 'Estonia',
        'Lettland' : 'Latvia',
        'Litauen' : 'Lithuania',
        'Saudi-Arabien' : 'Saudi Arabia',
        'Vereinigte Arabische Emirate' : 'United Arab Emirates',
        'Kroatien' : 'Croatia',
        'Rumänien' : 'Romania',
        'Russland' : 'Russia',
        'Slowakei' : 'Slovakia',
        'Slowenien' : 'Slovenia',
        'Tschechien' : 'Czech Republic',
        'Serbien' : 'Serbia',
        }
    
    for k in range(0, len(dict_regions)):
        for i in range(0, len(dict_regions[k][regions[k]])):
            if dict_regions[k][regions[k]].iloc[i]['Herkunftsland'] in trans_dict.keys():
                dict_regions[k][regions[k]].at[i,'Herkunftsland'] = trans_dict[dict_regions[k][regions[k]].iloc[i]['Herkunftsland']]
    trans_months = {
        'Januar' : 1,
        'Februar' : 2,
        'März' : 3,
        'April' : 4,
        'Mai' : 5,
        'Juni' : 6,
        'Juli' : 7,
        'August' : 8,
        'September' : 9,
        'Oktober' : 10,
        'November' : 11,
        'Dezember' : 12}
    for k in range(0, len(dict_regions)):
        for i in range(0, len(dict_regions[k][regions[k]])):
            if dict_regions[k][regions[k]].iloc[i]['month'] in trans_months.keys():
                dict_regions[k][regions[k]].at[i,'month'] = trans_months[dict_regions[k][regions[k]].iloc[i]['month']]
    for i in range(0, len(dict_regions)):
        dict_regions[i][regions[i]]['DATE'] = pd.to_datetime(dict_regions[i][regions[i]][['year','month']].assign(DAY = 1))
        dict_regions[i][regions[i]].drop(labels = ['year', 'month'], axis = 1, inplace = True)
    for i in range(0, len(dict_regions)):
        dict_regions[i][regions[i]] = dict_regions[i][regions[i]].pivot(index =  ['DATE'],columns = 'Herkunftsland', values = 'DATA').reset_index().rename_axis(None, axis=1)
     
    return(dict_regions, all_regions)

def gdp_clean (csv_data_url):
    gdp = pd.read_csv(f'{csv_data_url}')
    gdp[["q","year"]] = gdp["Period"].str.split("-",expand=True)
    del gdp["Period"]
    gdp.rename(columns={"Value":"value-usd", "LOCATION":"iso"}, inplace=True)
    gdp["year"] = gdp["year"].astype(int)
    gdp13 = gdp.loc[(gdp["year"] >= 2013)&(gdp["MEASURE"]=="HCPCARSA")]
    gdp13 = gdp13[["iso","year","value-usd","q"]]
    gdp13_wide = gdp13.pivot(index= ["year", "q"], columns="iso", values="value-usd").reset_index().rename_axis(None, axis=1)
    columns_old = gdp13_wide.columns.to_list()
    del columns_old[0:2]
    columns_new = [x + "_GDP" for x in columns_old]
    column_dict = dict(zip(columns_old,columns_new))
    gdp13_wide = gdp13_wide.rename(columns=column_dict)
    gdp13_wide_drop = gdp13_wide.drop(columns=["q","year"], axis=1)
    imputer = KNNImputer()
    dfimputed = imputer.fit_transform(gdp13_wide_drop)
    gdp13_wide_imp = pd.DataFrame(dfimputed)
    mice_imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), n_nearest_features=None, imputation_order='ascending')
    df_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(gdp13_wide_drop), columns=gdp13_wide_drop.columns)
    l = df_mice_imputed.columns.to_list()
    l1 = gdp13_wide.columns.to_list()
    list_int = [1,2,3,4,5,6,7,8,9,10,11,12]
    totallynewdf = pd.DataFrame()
    totallynewdf[l1] = pd.DataFrame(np.repeat(gdp13_wide.values, 3, axis=0))
    totallynewdf[l] = pd.DataFrame(np.repeat(gdp13_wide_imp.values, 3, axis=0))
    totallynewdf[["SAU_GDP","RUS_GDP"]] = pd.DataFrame(np.repeat(df_mice_imputed[["SAU_GDP","RUS_GDP"]].values, 3, axis=0))
    totallynewdf['q'] = np.tile(list_int, len(totallynewdf)//len(list_int) + 1)[:len(totallynewdf)]
    totallynewdf.rename(columns={"q":"month"}, inplace=True)
    totallynewdf['DATE'] = pd.to_datetime(totallynewdf[['year', 'month']].assign(DAY=1))
    totallynewdf.drop(["year","month"], axis=1, inplace=True)
    return(totallynewdf)

def forex_clean (csv_data_url):
    exchange_rate = pd.read_csv(f"{csv_data_url}", sep = ";", skiprows = 2)
    exchange_rate[["y","m"]] = exchange_rate["DATE"].str.split("-",expand=True)
    exchange_rate["y"] = exchange_rate["y"].astype(int)
    del exchange_rate["DATE"]
    del exchange_rate["D0"]
    exchange_rate = exchange_rate.loc[exchange_rate["y"]>=2013].reset_index()
    exchange_rate_wide = exchange_rate.pivot(index=["y", "m"], columns="D1", values="Value").reset_index().rename_axis(None, axis=1)
    colnamesexrate = exchange_rate_wide.columns.tolist()
    exchange_rate_wide = exchange_rate_wide.add_suffix('_exrate')
    exchange_rate_wide.rename(columns = {'y_exrate':'Year', 'm_exrate':'Month'}, inplace = True)
    exchange_rate_wide['DATE'] = pd.to_datetime(exchange_rate_wide[['Year', 'Month']].assign(DAY=1))
    exchange_rate_wide.drop([117], axis=0, inplace=True)
    del exchange_rate_wide["Year"]
    del exchange_rate_wide["Month"]
    return(exchange_rate_wide)

def ppi_clean(ppi_data_csv_url):
    PPI = pd.read_csv(f'{ppi_data_csv_url}')
    PPI[["y","m"]] = PPI["TIME"].str.split("-",expand=True)
    PPI["y"] = PPI["y"].astype(int)
    PPI.drop(labels = ['INDICATOR','SUBJECT','MEASURE','FREQUENCY','Flag Codes'], axis = 1, inplace = True)
    del PPI["TIME"]
    PPI = PPI.loc[PPI["y"]>=2013].reset_index()
    PPI_wide = PPI.pivot(index=["y", "m"], columns="LOCATION", values="Value").reset_index().rename_axis(None, axis=1)
    colnamesunemprate = PPI_wide.columns.tolist()
    PPI_wide = PPI_wide.add_suffix('_unemprate')
    PPI_wide.rename(columns = {'y_unemprate':'y', 'm_unemprate':'m'}, inplace = True)
    PPI_wide.drop([117], axis=0, inplace=True)
    PPI_wide["DATE"] = datelist 
    PPI_wide.drop(labels = ['y','m'], axis = 1, inplace = True)
    for i in PPI_wide.columns[PPI_wide.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
        PPI_wide[i].fillna(PPI_wide[i].mean(),inplace=True)
    return(PPI_wide)

def weather_clean(csv_url):
    xxx = pd.read_csv(csv_url, sep=";")
    weather_dict = {
        "tnd00xm0":"eistage",
        "tnd00nm0":"frosttage",
        "tnd30xm0":"hitzetage",
        "tre200m0":"avg. temp",
        "tre2dymx":"avg. maxtemp",
        "hns010mx":"cm neuschnee max10day",
        "hns000m0":"cm neuschnee",
        "rre150m0":"mm rain",
        "hto000m0":"cm avg. snowheight",
        "rsd010m0":"days rain >1mm",
        "rsd100m0":"days rain >10mm",
        "rs1000m0":"days rain >100mm",
        "stn":"standort",
        "time":"DATE"}
    xxx = xxx.rename(columns=weather_dict)
    if 117 in xxx.index:
        xxx.drop(index=[117], inplace=True)
        return(xxx)
    xxx["time"] = datelist 
    return(xxx)

def show_na(dfs):
    return(msno.matrix(dfs))
 
datelist.shape()