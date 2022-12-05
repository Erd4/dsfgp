# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 20:44:21 2022

@author: iphon
"""

# coding: utf-8

# In[5]:


import pandas as pd
import prophet as Prophet


# Importing modules for Prophet

# In[8]:


df = pd.read_csv("data_gp/total_df.csv").drop(columns=["Unnamed: 0"])
df["DATE"] = pd.to_datetime(df["DATE"])

#preparing df for prophet to work with
kol = df.columns.to_list()
add_regs_list=[]
for i in range(230):
    add_regs_list.append("add"+str(i+1))
add_regs_list
rename_df = dict(zip(kol,add_regs_list))
rename_df['DATE'] = "ds"
rename_df['Herkunftsland - Total'] = "y"

# renaming DF with new column names:
df.rename(columns={"DATE":"ds","Herkunftsland - Total":"y"},inplace=True)

#finally dropping all the country visitors
listen = ['Argentina',
 'Australien',
 'Austria',
 'Bahrain',
 'Belarus',
 'Belgium',
 'Brasil',
 'Bulgaria',
 'Canada',
 'China',
 'Croatia',
 'Cyprus',
 'Czech Republic',
 'Denmark',
 'Egypt',
 'Estonia',
 'Finland',
 'France',
 'Germany',
 'Greece',
 'Hongkong',
 'Hungary',
 'Iceland',
 'India',
 'Indonesia',
 'Ireland',
 'Israel',
 'Italy',
 'Japan',
 'Kuwait',
 'Latvia',
 'Liechtenstein',
 'Lithuania',
 'Luxembourg',
 'Malaysia',
 'Malta',
 'Mexico',
 'Netherlands',
 'New Zealand',
 'Norway',
 'Oman',
 'Philippinen',
 'Poland',
 'Portugal',
 'Qatar',
 'Romania',
 'Russia',
 'Saudi Arabia',
 'Serbia',
 'Singapore',
 'Slovakia',
 'Slovenia',
 'South Africa',
 'South Korea',
 'Spain',
 'Sweden',
 'Switzerland',
 'Taiwan',
 'Thailand',
 'Turkey',
 'Ukraine',
 'United Arab Emirates',
 'United Kingdom',
 'United States']

df.drop(columns=listen,inplace=True)


# Now we're going to simulate the first Prophet prediction without adding regressors

# In[10]:


# Fitting dataframe into Prophet
m = Prophet.Prophet(yearly_seasonality=True)
m.fit(df)


#making a new DF for our prediction for the future
#future = m.make_future_dataframe(periods=6, freq="MS")
future = m.make_future_dataframe(periods=6, freq = 'MS')
future.tail()

#getting prophet to forecast future guest numbers into the new future-df based on the data in our original df
#fcst = m.predict(future)
forecast_m = m.predict(future)


# Cross validating data and getting performance metrics

# In[11]:


from prophet.diagnostics import cross_validation
df_cv_m = cross_validation(m, horizon = '1y')


# Getting performance metrics

# In[12]:


import numpy as np
from prophet.diagnostics import performance_metrics
df_p_m = performance_metrics(df_cv_m)
m_mean_mae = np.mean(df_p_m['mae'])
m_mean_rmse = np.mean(df_p_m['rmse'])
m_mean_smape = np.mean(df_p_m['smape'])

# Plotting results

# In[13]:


from prophet.plot import plot_cross_validation_metric
ax1_m = m.plot(forecast_m, include_legend= True, xlabel = 'Year', ylabel = 'Visitors')
ax2_m = m.plot_components(forecast_m)
ax3_m = plot_cross_validation_metric(df_cv_m, metric='mae')


# Now trying to fit some exogenous variables and be a bit more precise so lets check the correlation first:

# In[14]:


df.corrwith(df["y"]).nsmallest(6)
df.corrwith(df["y"]).nlargest(9)


# As we can see we have some very high correlations like st.moritz visitors but also low ones like rainy days with not that much downfall,
# thus we are going to add our top 5 correlated regressors now that are not other communes.

# In[15]:


m2 = Prophet.Prophet(yearly_seasonality=True)
m2.add_regressor("cm avg. snowheight - chd", standardize=False)
m2.add_regressor('cm avg. snowheight - gsg', standardize=False)
m2.add_regressor('cm neuschnee - chd', standardize=False)
m2.add_regressor('eistage - chd', standardize=False)
m2.add_regressor('frosttage - chd', standardize=False)
m2.fit(df)


# Creating future dataframes for all the different regressors

# In[16]:


m_snow_chd = Prophet.Prophet(yearly_seasonality=True)
df_snow_chd = df.rename(columns = {'y' : 'bad', 'cm avg. snowheight - chd' : 'y'})
m_snow_chd.fit(df_snow_chd)
future_df_snow_chd = m_snow_chd.make_future_dataframe(periods=36, freq = 'MS')
forecast_snow_chd = m_snow_chd.predict(future_df_snow_chd)
forecast_snow_chd.rename(columns = {'yhat' : 'cm avg. snowheight - chd'}, inplace = True)

m_snow_gsg = Prophet.Prophet(yearly_seasonality=True)
df_snow_gsg = df.rename(columns = {'y' : 'bad', 'cm avg. snowheight - gsg' : 'y'})
m_snow_gsg.fit(df_snow_gsg)
future_df_snow_gsg = m_snow_gsg.make_future_dataframe(periods=36, freq = 'MS')
forecast_snow_gsg = m_snow_gsg.predict(future_df_snow_gsg)
forecast_snow_gsg.rename(columns = {'yhat' : 'cm avg. snowheight - gsg'}, inplace = True)

m_new_snow = Prophet.Prophet(yearly_seasonality=True)
df_new_snow = df.rename(columns = {'y' : 'bad', 'cm neuschnee - chd' : 'y'})
m_new_snow.fit(df_new_snow)
future_df_new_snow = m_new_snow.make_future_dataframe(periods = 36, freq = 'MS')
forecast_new_snow = m_new_snow.predict(future_df_new_snow)
forecast_new_snow.rename(columns = {'yhat' : 'cm neuschnee - chd'}, inplace = True)

m_ice = Prophet.Prophet(yearly_seasonality=True)
df_ice = df.rename(columns = {'y' : 'bad', 'eistage - chd' : 'y'})
m_ice.fit(df_ice)
future_df_ice = m_ice.make_future_dataframe(periods=36, freq = 'MS')
forecast_ice = m_ice.predict(future_df_ice)
forecast_ice.rename(columns = {'yhat' : 'eistage - chd'}, inplace = True)

m_frost = Prophet.Prophet(yearly_seasonality=True)
df_frost = df.rename(columns = {'y' : 'bad', 'frosttage - chd' : 'y'})
m_frost.fit(df_frost)
future_df_frost = m_frost.make_future_dataframe(periods=36, freq = 'MS')
forecast_frost = m_frost.predict(future_df_frost)
forecast_frost.rename(columns = {'yhat' : 'frosttage - chd'}, inplace = True)


# Creating future dataframe for our main dataframe and merging it with the different regressors.

# In[17]:


future_df = m2.make_future_dataframe(periods=36, freq = 'MS')

future_df = pd.merge(future_df, forecast_frost[['frosttage - chd', 'ds']], on = 'ds', how = 'inner')
future_df = pd.merge(future_df, forecast_ice[['eistage - chd', 'ds']], on = 'ds', how = 'inner')
future_df = pd.merge(future_df, forecast_new_snow[['cm neuschnee - chd', 'ds']], on = 'ds', how = 'inner')
future_df = pd.merge(future_df, forecast_snow_gsg[['cm avg. snowheight - gsg', 'ds']], on = 'ds', how = 'inner')
future_df = pd.merge(future_df, forecast_snow_chd[['cm avg. snowheight - chd', 'ds']], on = 'ds', how = 'inner')


# Forecasting our dataframe with our different regressors.

# In[18]:


forecast_m2 = m2.predict(future_df)


# Cross validating

# In[19]:


df_cv_m2 = cross_validation(m2, horizon = '1y')


# Getting our different metrics to compare old prediction with new prediction, namely functions m and m2.

# In[20]:


df_p_m2 = performance_metrics(df_cv_m2)
m2_mean_mae = np.mean(df_p_m2['mae'])
m2_mean_rmse = np.mean(df_p_m2['rmse'])
m2_mean_smape = np.mean(df_p_m2['smape'])
# Plotting the new prediction

# In[21]:


ax1_m2 = m2.plot(forecast_m2, include_legend= True, xlabel = 'Year', ylabel = 'Visitors')
ax2_m2 = m2.plot_components(forecast_m2)
ax3_m2 = plot_cross_validation_metric(df_cv_m2, metric='mae')


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize = (12,6), dpi = 1000)
sns.lineplot(data = df_p_m2['rmse'], ax = axs[0,0]).axhline(np.mean(df_p_m2['rmse']), c='red', linestyle='dashed', label="Mean")
sns.lineplot(data = df_p_m2['mae'], ax = axs[1,0]).axhline(np.mean(df_p_m2['mae']), c='red', linestyle='dashed')
sns.lineplot(data = df_p_m2['smape'], ax = axs[2,0]).axhline(np.mean(df_p_m2['smape']), c='red', linestyle='dashed')
sns.lineplot(data = df_p_m['rmse'], ax = axs[0,1]).axhline(np.mean(df_p_m['rmse']), c='red', linestyle='dashed')
sns.lineplot(data = df_p_m['mae'], ax = axs[1,1]).axhline(np.mean(df_p_m['mae']), c='red', linestyle='dashed')
sns.lineplot(data = df_p_m['smape'], ax = axs[2,1]).axhline(np.mean(df_p_m['smape']), c='red', linestyle='dashed')
axs[0,0].set_title('Prediction with Regressors')
axs[0,1].set_title('Prediction without Regressors')
axs[0,0].set_ylim(2400, 7700)
axs[0,1].set_ylim(2400, 7700)
axs[1,0].set_ylim(1700, 5500)
axs[1,1].set_ylim(1700, 5500)
axs[2,0].set_ylim(0.05, 0.47)
axs[2,1].set_ylim(0.05, 0.47)
fig.legend(loc = "lower center")

# In[30]:

df_bar = pd.DataFrame(index = [0])

df_bar['m2_mean_rmse'] = m2_mean_rmse
df_bar['m_mean_rmse'] = m_mean_rmse
df_bar['m2_mean_mae'] = m2_mean_mae
df_bar['m_mean_mae'] = m_mean_mae
df_bar['m2_mean_smape'] = m2_mean_smape
df_bar['m_mean_smape'] = m_mean_smape

fig, ax_bar = plt.subplots(1,3,figsize = (12, 6), dpi = 1000)
palette = sns.color_palette("Paired")
sns.barplot(data = df_bar[['m2_mean_rmse','m_mean_rmse']], ax = ax_bar[0], palette = palette).set(xticklabels=['With Regressors', 'Without Regressors']) 
sns.barplot(data = df_bar[['m2_mean_mae', 'm_mean_mae']], ax = ax_bar[1], palette = palette).set(xticklabels=['With Regressors', 'Without Regressors']) 
sns.barplot(data = df_bar[['m2_mean_smape', 'm_mean_smape']], ax = ax_bar[2], palette = palette).set(xticklabels=['With Regressors', 'Without Regressors']) 
ax_bar[0].set_title('Mean of Root-Mean-Squared Error')
ax_bar[1].set_title('Mean of Mean Absolute Error')
ax_bar[2].set_title('Mean of Symmetric Mean Absolute Percentage Error')
fig.suptitle('Various Metrics for Prophet Predictions with and without regressors')
# Using a second model, namely SARIMAX, to predict the visitors in the region

# In[9]:

import pandas as pd
df_max = pd.read_csv("data_gp/total_df.csv").drop(columns=["Unnamed: 0"])
df_max["DATE"] = pd.to_datetime(df_max["DATE"])
df_max = df_max.set_index("DATE")


# Preparing our dataframe and defining exogenous variables as well as our endogenous variable,
# and creating array with n observations of variables and k number of variables and then filling it.

# In[10]:


exog = df_max.drop(columns=listen).to_numpy()
exo = df_max.drop(columns=listen)
endo = df_max["Herkunftsland - Total"]

from statsmodels.tsa.seasonal import seasonal_decompose

df_seasonal = seasonal_decompose(endo, model='additive')

#It's important to see a strong seasonal influence - otherwise we use ARIMA(X)
plt.rc('figure',figsize=(14,8))
plt.rc('font',size=15)
df_seasonal.plot();
plt.show()

# Plotting endogenous variables to see if it's stationary, and it actually seems

# In[11]:

import seaborn as sns
sns.lineplot(endo)


# But to be really sure we can check it's stationarity with an augmented dickey-fuller-test

# In[17]:


import predhelp as ph
ph.check_stationarity(endo)


# Freude herrscht! Apparently it's not - as if it wasn't hard enough already.
# *at least we now know that d is not equal to 0 - but i'll get to that later*
# Let's try to make endo stationary by removing the trend:
# we try to do this by taking the difference of between the current value and the prior month's value.

# In[14]:


import numpy as np
endo_diff = endo.diff()[1:] 
exog_diff = np.diff(exog)[1:]
#sns.lineplot(endo_diff)


# Also Dickey-Fuller agrees with us.

# In[ ]:


ph.check_stationarity(endo_diff)


# Cross Validation by using sklearn TimeSeiresSplit

# In[15]:


from sklearn.model_selection import TimeSeriesSplit

tss = TimeSeriesSplit(n_splits=6)

for train_index, test_index in tss.split(exog):
    exog_train, exog_test = exog[train_index, :], exog[test_index,:]
    endo_train, endo_test = endo.iloc[train_index], endo.iloc[test_index]
    exo_train, exo_test = exo.iloc[train_index], exo.iloc[test_index]


# In[17]:


endo_train.groupby('DATE').mean().plot()
endo_test.groupby('DATE').mean().plot()



# In[ ]:


from datetime import datetime
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX


train_end = datetime(2021,5,1)
test_end = datetime(2022,9,1)
startparams = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1]
traintrain_exo = exo[train_end:]

rolling_predictions = endo_test.copy()
for train_end in endo_test.index:
    train_data = endo[:train_end-timedelta(days=1)]
    train_exo = exo[:train_end-timedelta(days=1)]
    model = SARIMAX(train_data, order=(1,2,0), seasonal_order=(0,2,1,12),exog=train_exo, )
    model_fit = model.fit(start_params=None)
    
    pred = model_fit.forecast(exog=traintrain_exo.loc[[train_end-timedelta()]])
    rolling_predictions[train_end] = pred


# In[ ]:



# In[26]:


rolling_residuals = endo_test - rolling_predictions


# In[28]:

import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(rolling_residuals)
plt.axhline(0, linestyle='--', color='k')
plt.title('Rolling Forecast Residuals from SARIMA Model', fontsize=20)
plt.ylabel('Error', fontsize=16)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize = (12,6), dpi = 1000)
ax.plot(rolling_predictions, label="Pred")
ax.plot(endo, label="True")
ax.set_title("Number of guests in the tourism region of Gstaad", fontsize=20)
ax.set_ylabel('Visitors', fontsize=16) 
ax.legend()


# In[34]:


rolling_predictions


# Getting the different metrics of our SARIMAX prediction

# In[1]:


pred.mae()


# In[ ]:


def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

smape(endo_test, rolling_predictions)


from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_error

rmse(endo_test, rolling_predictions)
mean_absolute_error(endo_test, rolling_predictions)

