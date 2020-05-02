#!/usr/bin/env python
# coding: utf-8

# In[496]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts 
import seaborn as sns
sns.set()
from arch import arch_model
import yfinance
from pmdarima.arima import auto_arima


# In[15]:


raw_csv_data = pd.read_csv("C:/Users/Admin/Desktop/Index2018.csv") 
df_comp=raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index("date", inplace=True)
df_comp=df_comp.asfreq('b')
df_comp=df_comp.fillna(method='ffill')


# In[16]:


df_comp['market_value']=df_comp.ftse


# In[17]:


del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]


# In[18]:


sgt.plot_acf(df.market_value, zero = False, lags = 40)
plt.title("ACF for Prices", size = 20)
plt.show()


# In[19]:


import warnings
warnings.filterwarnings("ignore")


# In[20]:


sgt.plot_pacf(df.market_value, lags = 40, alpha = 0.05, zero = False, method = ('ols'))
plt.title("PACF for Prices", size = 20)
plt.show()


# In[21]:


model_ar = ARMA(df.market_value, order=(1,0))
results_ar = model_ar.fit()
results_ar.summary()


# In[22]:


model_ar_2 = ARMA(df.market_value, order=(2,0))
results_ar_2 = model_ar_2.fit()
results_ar_2.summary()


# In[23]:


model_ar_3 = ARMA(df.market_value, order=(3,0))
results_ar_3 = model_ar_3.fit()
results_ar_3.summary()


# In[24]:


model_ar_4 = ARMA(df.market_value, order=[4,0])
results_ar_4 = model_ar_4.fit()
results_ar_4.summary()


# In[25]:


def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR, DF).round(3)
    return p


# In[26]:


LLR_test(model_ar_2, model_ar_3)


# In[27]:


LLR_test(model_ar_3, model_ar_4)


# In[29]:


model_ar_4 = ARMA(df.market_value, order=[4,0])
results_ar_4 = model_ar_4.fit()
print(results_ar_4.summary()) 
print ("LLR test: " + str(LLR_test(model_ar_3, model_ar_4)))


# In[31]:


model_ar_5 = ARMA(df.market_value, order=(5,0))
results_ar_5 = model_ar_5.fit()
print(results_ar_5.summary()) 
print ("LLR test: " + str(LLR_test(model_ar_4, model_ar_5)))


# In[33]:


model_ar_6 = ARMA(df.market_value, order=[6,0])
results_ar_6 = model_ar_4.fit()
print(results_ar_6.summary()) 
print ("LLR test: " + str(LLR_test(model_ar_5, model_ar_6)))


# In[35]:


model_ar_7 = ARMA(df.market_value, order=[7,0])
results_ar_7 = model_ar_7.fit()
print(results_ar_7.summary()) 
print ("LLR test: " + str(LLR_test(model_ar_6, model_ar_7)))


# In[37]:


model_ar_8 = ARMA(df.market_value, order=[8,0])
results_ar_8 = model_ar_8.fit()
print(results_ar_8.summary()) 
print ("LLR test: " + str(LLR_test(model_ar_7, model_ar_8)))


# In[39]:


print("LLR test: " + str(LLR_test(model_ar, model_ar_7, DF = 6)))


# In[54]:


df.head()


# In[44]:


sts.adfuller(df.market_value)


# In[45]:


df['returns'] = df.market_value.pct_change(1).mul(100)


# In[49]:


df = df.iloc[1 :]


# In[50]:


sts.adfuller(df.returns)


# In[53]:


df.head()


# In[56]:


sgt.plot_acf(df.returns, lags = 40, zero = False)
plt.title("ACF vs FTSE Returns", size = 24)
plt.show()


# In[57]:


sgt.plot_pacf(df.returns, lags = 40, zero = False)
plt.title("PACF vs FTSE Retunrs", size = 24)
plt.show()


# In[65]:


model_ret_ar_1 = ARMA(df.returns, order=(1,0))
results_ret_ar_1 = model_ret_ar_1.fit()
print(results_ret_ar_1.summary())


# In[66]:


model_ret_ar_2 = ARMA(df.returns, order=(2,0))
results_ret_ar_2 = model_ret_ar_2.fit()
print(results_ret_ar_2.summary())
print ("LLR test: " + str(LLR_test(model_ret_ar_1, model_ret_ar_2)))


# In[68]:


model_ret_ar_3 = ARMA(df.returns, order=(3,0))
results_ret_ar_3 = model_ret_ar_3.fit()
print(results_ret_ar_3.summary())
print ("LLR test: " + str(LLR_test(model_ret_ar_2, model_ret_ar_3)))


# In[70]:


model_ret_ar_4 = ARMA(df.returns, order=(4,0))
results_ret_ar_4 = model_ret_ar_4.fit()
print(results_ret_ar_4.summary())
print ("LLR test: " + str(LLR_test(model_ret_ar_3, model_ret_ar_4)))


# In[72]:


model_ret_ar_5 = ARMA(df.returns, order=(5,0))
results_ret_ar_5 = model_ret_ar_5.fit()
print(results_ret_ar_5.summary())
print ("LLR test: " + str(LLR_test(model_ret_ar_4, model_ret_ar_5)))


# In[74]:


model_ret_ar_6 = ARMA(df.returns, order=(6,0))
results_ret_ar_6 = model_ret_ar_6.fit()
print(results_ret_ar_6.summary())
print ("LLR test: " + str(LLR_test(model_ret_ar_5, model_ret_ar_6)))


# In[76]:


model_ret_ar_7 = ARMA(df.returns, order=(7,0))
results_ret_ar_7 = model_ret_ar_7.fit()
print(results_ret_ar_7.summary())
print ("LLR test: " + str(LLR_test(model_ret_ar_6, model_ret_ar_7)))


# In[79]:


benchmark = df.market_value.iloc[0]
df['norm'] = df.market_value.div(benchmark).mul(100)


# In[81]:


df.head()


# In[83]:


sts.adfuller(df.norm)


# In[86]:


bench_ret = df.returns.iloc[0]
df['norm_ret'] = df.returns.div(bench_ret).mul(100)


# In[88]:


sts.adfuller(df.norm_ret)


# In[90]:


df.head()


# In[93]:


model_norm_ret_ar_1 = ARMA(df.norm_ret, order= (1, 0))
results_norm_ret_ar_1 = model_norm_ret_ar_1.fit()
results_norm_ret_ar_1.summary()


# In[98]:


model_norm_ret_ar_2 = ARMA(df.norm_ret, order= (2, 0))
results_norm_ret_ar_2 = model_norm_ret_ar_2.fit()
results_norm_ret_ar_2.summary()


# In[99]:


print ("LLR test: " + str(LLR_test(model_norm_ret_ar_1, model_norm_ret_ar_2)))


# In[100]:


model_norm_ret_ar_3 = ARMA(df.norm_ret, order= (3, 0))
results_norm_ret_ar_3 = model_norm_ret_ar_3.fit()
results_norm_ret_ar_3.summary()


# In[101]:


model_norm_ret_ar_4 = ARMA(df.norm_ret, order= (4, 0))
results_norm_ret_ar_4 = model_norm_ret_ar_4.fit()
results_norm_ret_ar_4.summary()


# In[102]:


model_norm_ret_ar_5 = ARMA(df.norm_ret, order= (5, 0))
results_norm_ret_ar_5 = model_norm_ret_ar_5.fit()
results_norm_ret_ar_5.summary()


# In[103]:


model_norm_ret_ar_6 = ARMA(df.norm_ret, order= (6, 0))
results_norm_ret_ar_6 = model_norm_ret_ar_6.fit()
results_norm_ret_ar_6.summary()


# In[104]:


model_norm_ret_ar_7 = ARMA(df.norm_ret, order= (7, 0))
results_norm_ret_ar_7 = model_norm_ret_ar_7.fit()
results_norm_ret_ar_7.summary()


# In[106]:


model_norm_ret_ar_8 = ARMA(df.norm_ret, order= (8, 0))
results_norm_ret_ar_8 = model_norm_ret_ar_8.fit()
results_norm_ret_ar_8.summary()


# In[109]:


df['res_price'] = results_ar_7.resid
df.res_price.mean()


# In[112]:


df.res_price.var()


# In[114]:


sts.adfuller(df.res_price)


# In[116]:


sgt.plot_acf(df.res_price, lags = 40, zero = False)
plt.title("ACF of Residuals for prices", size = 24)
plt.show()


# In[119]:


df.res_price[1:].plot(figsize = (20,5))
plt.title('Residuals of Prices', Size = 24)
plt.show()


# In[121]:


df['res_ret'] = results_ret_ar_7.resid


# In[125]:


df.res_ret.mean()


# In[127]:


df.res_ret.var()


# In[129]:


sts.adfuller(df.res_ret)


# In[133]:


sgt.plot_acf(df.res_ret, lags =40, zero = False)
plt.title("ACF of Residuals for Returns", size = 24)
plt.show()


# In[135]:


df.res_ret.plot(figsize = (20,5))
plt.title("Residuals of Returns", size = 24)
plt.show()


# In[137]:


sgt.plot_acf(df.returns, lags = 40, zero = False)
plt.title("ACF of resturns", size = 24)
plt.show()


# In[139]:


df.head()


# In[141]:


df['returns'] = df.market_value.pct_change(1)*100


# In[143]:


df.head()


# In[146]:


sgt.plot_acf(df.returns.iloc[1:], lags = 40 , zero = False)
plt.title("ACF of Returns", size =24)
plt.show()


# In[148]:


model_ret_ma_1 = ARMA(df.returns[1:], order = (0,1))
results_ret_ma_1 = model_ret_ma_1.fit()
results_ret_ma_1.summary()


# In[153]:


model_ret_ma_2 = ARMA(df.returns[1:], order = (0,2))
results_ret_ma_2 = model_ret_ma_2.fit()
print(results_ret_ma_2.summary())
print("\n LLR test P value = : " + str(LLR_test(model_ret_ma_1, model_ret_ma_2)))


# In[154]:


model_ret_ma_3 = ARMA(df.returns[1:], order = (0,3))
results_ret_ma_3 = model_ret_ma_3.fit()
print(results_ret_ma_3.summary())
print("\n LLR test P value = : " + str(LLR_test(model_ret_ma_2, model_ret_ma_3)))


# In[155]:


model_ret_ma_4 = ARMA(df.returns[1:], order = (0,4))
results_ret_ma_4 = model_ret_ma_4.fit()
print(results_ret_ma_4.summary())
print("\n LLR test P value = : " + str(LLR_test(model_ret_ma_3, model_ret_ma_4)))


# In[156]:


model_ret_ma_5 = ARMA(df.returns[1:], order = (0,5))
results_ret_ma_5 = model_ret_ma_5.fit()
print(results_ret_ma_5.summary())
print("\n LLR test P value = : " + str(LLR_test(model_ret_ma_4, model_ret_ma_5)))


# In[157]:


model_ret_ma_6 = ARMA(df.returns[1:], order = (0,6))
results_ret_ma_6 = model_ret_ma_6.fit()
print(results_ret_ma_6.summary())
print("\n LLR test P value = : " + str(LLR_test(model_ret_ma_5, model_ret_ma_6)))


# In[158]:


model_ret_ma_7 = ARMA(df.returns[1:], order = (0,7))
results_ret_ma_7 = model_ret_ma_7.fit()
print(results_ret_ma_7.summary())
print("\n LLR test P value = : " + str(LLR_test(model_ret_ma_6, model_ret_ma_7)))


# In[164]:


model_ret_ma_8 = ARMA(df.returns[1:], order = (0,8))
results_ret_ma_8 = model_ret_ma_8.fit()
print(results_ret_ma_8.summary())
print("\n LLR test P value = : " + str(LLR_test(model_ret_ma_7, model_ret_ma_8)))


# In[165]:


LLR_test(model_ret_ma_6, model_ret_ma_8, DF = 2)


# In[170]:


df['res_ret_ma_8'] = results_ret_ma_8.resid[1:]


# In[176]:


round(df.res_ret_ma_8.mean(),4)


# In[179]:


round(df.res_ret_ma_8.var(), 3)


# In[193]:


df.res_ret_ma_8[1:].plot(figsize = (20,5))
plt.title("Residuals of MA Returns", size = 24)
plt.show()


# In[207]:


df['res_ret_ma_8']=df.res_ret_ma_8.fillna(method= 'bfill')


# In[210]:


df.res_ret_ma_8.isna().sum()


# In[221]:


sts.adfuller(df.res_ret_ma_8[2:])


# In[215]:


sgt.plot_acf(df.res_ret_ma_8[2:], lags = 40, zero = False)
plt.title("ACF od Residuals of Returns", size = 24)
plt.show()


# In[217]:


df.head()


# In[223]:


bench_ret = df.returns.iloc[1]
df['norm_ret']= df.returns.div(bench_ret).mul(100)


# In[224]:


sgt.plot_acf(df.norm_ret[1:], lags = 40, zero = False)
plt.title("ACF of Normalized Returns", size = 24)
plt.show()


# In[227]:


model_norm_ret_ma_8 = ARMA(df.norm_ret[1:], order =(0,8))
results_norm_ret_ma_8 = model_norm_ret_ma_8.fit()
results_norm_ret_ma_8.summary()


# In[229]:


df['res_norm_ret_ma_8'] = results_ret_ma_8.resid[1:]


# In[231]:


df.res_norm_ret_ma_8.plot(figsize = (20,5))
plt.title("Residuals of Normalized Returns", size = 24)
plt.show()


# In[233]:


sgt.plot_acf(df.res_norm_ret_ma_8[2:], lags = 40, zero = False)
plt.title("ACF of Residuals of Normalized Returns", size = 24)
plt.show()


# In[236]:


sgt.plot_acf(df.market_value, lags = 40, zero = False)
plt.title('ACF for Prices', size = 24)
plt.show()


# In[238]:


model_ma_1 = ARMA(df.market_value, order = (0,1))
results_model_ma_1 = model_ma_1.fit()
results_model_ma_1.summary()


# In[250]:


model_ret_ar_1_ma_1 = ARMA(df.returns[1:], order = (1,1))
results_ret_ar_1_ma_1 = model_ret_ar_1_ma_1.fit()
results_ret_ar_1_ma_1.summary()


# In[245]:


model_ret_ar_2_ma_2 = ARMA(df.returns[1:], order=(2,2))
results_ret_ar_2_ma_2 = model_ret_ar_2_ma_2.fit()
results_ret_ar_2_ma_2.summary()


# In[248]:


model_ret_ar_3_ma_3 = ARMA(df.returns[1:], order=(3,3))
results_ret_ar_3_ma_3 = model_ret_ar_3_ma_3.fit()
results_ret_ar_3_ma_3.summary()


# In[252]:


LLR_test(model_ret_ar_1_ma_1, model_ret_ar_3_ma_3, DF = 4)


# In[254]:


model_ret_ar_3_ma_2 = ARMA(df.returns[1:], order=(3,2))
results_ret_ar_3_ma_2 = model_ret_ar_3_ma_2.fit()
results_ret_ar_3_ma_2.summary()


# In[256]:


model_ret_ar_2_ma_3 = ARMA(df.returns[1:], order=(2,3))
results_ret_ar_2_ma_3 = model_ret_ar_2_ma_3.fit()
results_ret_ar_2_ma_3.summary()


# In[274]:


model_ret_ar_3_ma_1 = ARMA(df.returns[1:], order=(3,1))
results_ret_ar_3_ma_1 = model_ret_ar_3_ma_1.fit()
print(results_ret_ar_3_ma_1.summary())
print("\n LLR test for p value = :" + str(LLR_test(model_ret_ar_3_ma_1, model_ret_ar_3_ma_2)))


# In[261]:


model_ret_ar_1_ma_3 = ARMA(df.returns[1:], order=(1,3))
results_ret_ar_1_ma_3 = model_ret_ar_1_ma_3.fit()
print(results_ret_ar_1_ma_3.summary())
print("\n LLR test for p value = :" + str(LLR_test(model_ret_ar_1_ma_3, model_ret_ar_2_ma_2)))


# In[263]:


LLR_test(model_ret_ar_3_ma_2, model_ret_ar_1_ma_3)


# In[267]:


print(results_ret_ar_3_ma_2.llf, results_ret_ar_1_ma_3.aic)
print(results_ret_ar_1_ma_3.llf, results_ret_ar_1_ma_3.aic)


# In[269]:


df['res_ret_ar_3_ma_2'] = results_ret_ar_3_ma_2.resid[1:]


# In[272]:


df.res_ret_ar_3_ma_2.plot(figsize = (20,5))
plt.title("Residuals of Returns", size = 24)
plt.show()


# In[275]:


sgt.plot_acf(df.res_ret_ar_3_ma_2, lags = 40, zero =False)
plt.title("ACF for Residuals for Returns ", size = 24)
plt.show()


# In[277]:


model_ret_ar_1_ma_5 = ARMA(df.returns[1:], order=(1,5))
results_ret_ar_1_ma_5 = model_ret_ar_1_ma_5.fit()
results_ret_ar_1_ma_5.summary()


# In[279]:


model_ret_ar_2_ma_5 = ARMA(df.returns[1:], order=(2,5))
results_ret_ar_2_ma_5 = model_ret_ar_2_ma_5.fit()
results_ret_ar_2_ma_5.summary()


# In[281]:


model_ret_ar_3_ma_5 = ARMA(df.returns[1:], order=(3,5))
results_ret_ar_3_ma_5 = model_ret_ar_3_ma_5.fit()
results_ret_ar_3_ma_5.summary()


# In[283]:


model_ret_ar_4_ma_5 = ARMA(df.returns[1:], order=(4,5))
results_ret_ar_4_ma_5 = model_ret_ar_4_ma_5.fit()
results_ret_ar_4_ma_5.summary()


# In[285]:


model_ret_ar_5_ma_1 = ARMA(df.returns[1:], order=(5,1))
results_ret_ar_5_ma_1 = model_ret_ar_5_ma_1.fit()
results_ret_ar_5_ma_1.summary()


# In[287]:


model_ret_ar_5_ma_2 = ARMA(df.returns[1:], order=(5,2))
results_ret_ar_5_ma_2 = model_ret_ar_5_ma_2.fit()
results_ret_ar_5_ma_2.summary()


# In[289]:


model_ret_ar_5_ma_3 = ARMA(df.returns[1:], order=(5,3))
results_ret_ar_5_ma_3 = model_ret_ar_5_ma_3.fit()
results_ret_ar_5_ma_3.summary()


# In[291]:


model_ret_ar_5_ma_4 = ARMA(df.returns[1:], order=(5,4))
results_ret_ar_5_ma_4 = model_ret_ar_5_ma_4.fit()
results_ret_ar_5_ma_4.summary()


# In[293]:


model_ret_ar_5_ma_5 = ARMA(df.returns[1:], order=(5,5))
results_ret_ar_5_ma_5 = model_ret_ar_5_ma_5.fit()
results_ret_ar_5_ma_5.summary()


# In[295]:


print(results_ret_ar_5_ma_1.llf, results_ret_ar_1_ma_5.aic)


# In[296]:


print(results_ret_ar_3_ma_2.llf, results_ret_ar_1_ma_3.aic)


# In[298]:


print(results_ret_ar_3_ma_2.llf, results_ret_ar_3_ma_2.aic)


# In[301]:


df['res_ret_ar_5_ma_1'] = results_ret_ar_5_ma_1.resid[1:]


# In[304]:


sgt.plot_acf(df.res_ret_ar_5_ma_1, lags = 40 , zero = False)
plt.title("ACF of Residuals for Returns", size = 24)
plt.show()


# In[307]:


sgt.plot_acf(df.market_value, lags = 40, zero = False, unbiased=True)
plt.title("ACF for Prices", size = 24)
plt.show()


# In[309]:


sgt.plot_pacf(df.market_value, lags = 40, zero = False, alpha = 0.05, method = 'ols')
plt.title("PACF for Prices", size = 24)
plt.show()


# In[311]:


df['res_ret_ar_1_ma_1'] = results_ret_ar_1_ma_1.resid[1:]


# In[314]:


sgt.plot_acf(df.res_ret_ar_1_ma_1, lags = 40, zero = False, unbiased=True)
plt.title("ACF for Residuals", size = 24)
plt.show()


# In[343]:


model_ar_6_ma_6 = ARMA(df.market_value, order=(6,6))
results_ar_6_ma_6 = model_ar_6_ma_6.fit(start_ar_lags = 11)
results_ar_6_ma_6.summary()


# In[344]:


model_ar_5_ma_6 = ARMA(df.market_value, order=(5,6))
results_ar_5_ma_6 = model_ar_5_ma_6.fit(start_ar_lags = 7)
results_ar_5_ma_6.summary()


# In[345]:


model_ar_6_ma_1 = ARMA(df.market_value, order=(6,1))
results_ar_6_ma_1 = model_ar_6_ma_1.fit(start_ar_lags = 7)
results_ar_6_ma_1.summary()


# In[327]:


print(results_ret_ar_5_ma_6.llf, results_ret_ar_5_ma_6.aic)


# In[329]:


print(results_ret_ar_6_ma_1.llf, results_ret_ar_6_ma_1.aic)


# In[348]:


df["res_ar_5_ma_6"] = results_ar_5_ma_6.resid


# In[353]:


sgt.plot_acf(df.res_ar_5_ma_6, lags = 40, zero = False)
plt.title("ACF for Residuals Price", size = 24)
plt.show()


# In[354]:


print("ARMA(5,6):  \t LL = ", results_ar_5_ma_6.llf, "\t AIC = ", results_ar_5_ma_6.aic)
print("ARMA(5,1):  \t LL = ", results_ret_ar_5_ma_1.llf, "\t AIC = ", results_ret_ar_5_ma_1.aic)


# In[372]:


model_ar_1_i_1_ma_1 = ARIMA(df.market_value, order = (1,1,1))
results_ar_1_i_1_ma_1 = model_ar_1_i_1_ma_1.fit()
results_ar_1_i_1_ma_1.summary()


# In[373]:


df['res_ar_1_i_1_ma_1'] = results_ar_1_i_1_ma_1.resid


# In[374]:


sgt.plot_acf(df.res_ar_1_i_1_ma_1, lags = 40 , zero = False)
plt.title("ACF for Residuals for ARIMA(1,1,1)", size = 24)
plt.show()


# In[375]:


df['res_ar_1_i_1_ma_1'] = results_ar_1_i_1_ma_1.resid.iloc[:]
sgt.plot_acf(df.res_ar_1_i_1_ma_1, lags = 40 , zero = False)
plt.title("ACF for Residuals for ARIMA(1,1,1)", size = 24)
plt.show()


# In[371]:


model_ar_1_i_1_ma_2 = ARIMA(df.market_value, order = (1,1,2))
results_ar_1_i_1_ma_2 = model_ar_1_i_1_ma_2.fit()
model_ar_1_i_1_ma_3 = ARIMA(df.market_value, order = (1,1,3))
results_ar_1_i_1_ma_3 = model_ar_1_i_1_ma_3.fit()
model_ar_2_i_1_ma_1 = ARIMA(df.market_value, order = (2,1,1))
results_ar_2_i_1_ma_1 = model_ar_2_i_1_ma_1.fit()
model_ar_3_i_1_ma_1 = ARIMA(df.market_value, order = (3,1,1))
results_ar_3_i_1_ma_1 = model_ar_3_i_1_ma_1.fit()
model_ar_3_i_1_ma_2 = ARIMA(df.market_value, order = (3,1,2))
results_ar_3_i_1_ma_2 = model_ar_3_i_1_ma_2.fit(start_ar_lags = 5)


# In[377]:


print("ARIMA(1,1,1):  \t LL = ", results_ar_1_i_1_ma_1.llf, "\t AIC = ", results_ar_1_i_1_ma_1.aic)
print("ARIMA(1,1,2):  \t LL = ", results_ar_1_i_1_ma_2.llf, "\t AIC = ", results_ar_1_i_1_ma_2.aic)
print("ARIMA(1,1,3):  \t LL = ", results_ar_1_i_1_ma_3.llf, "\t AIC = ", results_ar_1_i_1_ma_3.aic)
print("ARIMA(2,1,1):  \t LL = ", results_ar_2_i_1_ma_1.llf, "\t AIC = ", results_ar_2_i_1_ma_1.aic)
print("ARIMA(3,1,1):  \t LL = ", results_ar_3_i_1_ma_1.llf, "\t AIC = ", results_ar_3_i_1_ma_1.aic)
print("ARIMA(3,1,2):  \t LL = ", results_ar_3_i_1_ma_2.llf, "\t AIC = ", results_ar_3_i_1_ma_2.aic)


# In[381]:


print("\nLLR test p-value = " + str (LLR_test(model_ar_1_i_1_ma_2, model_ar_1_i_1_ma_3)))


# In[396]:


print("\nLLR test p-value = " + str (LLR_test(model_ar_1_i_1_ma_2, model_ar_1_i_1_ma_3, DF = 2)))


# In[388]:


df['res_ar_1_i_1_ma_3'] = results_ar_1_i_1_ma_3.resid
sgt.plot_acf(df.res_ar_1_i_1_ma_3[1:], lags = 40, zero = False)
plt.title("ACF for Residuals for ARIMA(1,1,3)", size = 24)
plt.show()


# In[390]:


model_ar_5_i_1_ma_1 = ARIMA(df.market_value, order = (5,1,1))
results_ar_5_i_1_ma_1 = model_ar_5_i_1_ma_1.fit()
model_ar_6_i_1_ma_3 = ARIMA(df.market_value, order = (6,1,3))
results_ar_6_i_1_ma_3 = model_ar_6_i_1_ma_3.fit()


# In[392]:


results_ar_5_i_1_ma_1.summary()


# In[394]:


results_ar_6_i_1_ma_3.summary()


# In[402]:


print("ARIMA(1,1,3):  \t LL = ", results_ar_1_i_1_ma_3.llf, "\t AIC = ", results_ar_1_i_1_ma_3.aic)
print("ARIMA(5,1,1):  \t LL = ", results_ar_5_i_1_ma_1.llf, "\t AIC = ", results_ar_5_i_1_ma_1.aic)
print("ARIMA(6,1,3):  \t LL = ", results_ar_6_i_1_ma_3.llf, "\t AIC = ", results_ar_6_i_1_ma_3.aic)


# In[407]:


print("\nLLR test p-value = " + str (LLR_test(model_ar_1_i_1_ma_3, model_ar_1_i_1_ma_3, DF = 5)))


# In[406]:


print("\nLLR test p-value = " + str (LLR_test(model_ar_5_i_1_ma_1, model_ar_5_i_1_ma_1, DF = 3)))


# In[408]:


print("\nLLR test p-value = " + str (LLR_test(model_ar_6_i_1_ma_3, model_ar_6_i_1_ma_3)))


# In[410]:


df['res_ar_5_i_1_ma_1'] = results_ar_5_i_1_ma_1.resid


# In[412]:


sgt.plot_acf(df.res_ar_5_i_1_ma_1[1:], lags =40 , zero = False)
plt.title("ACF for Residuals for ARIMA(5,1,1)", size = 24)
plt.show()


# In[414]:


df['delta_prices'] = df.market_value.diff(1)


# In[416]:


model_delta_ar_1_i_1_ma_1 =ARIMA(df.delta_prices[1:], order = (1,0,1) )
results_delta_ar_1_i_1_ma_1 = model_delta_ar_1_i_1_ma_1.fit()
results_delta_ar_1_i_1_ma_1.summary()


# In[417]:


sts.adfuller(df.delta_prices[1:])


# In[419]:


model_ar_1_i_2_ma_1 = ARIMA(df.market_value, order = (1,2,1))
results_ar_1_i_2_ma_1 = model_ar_1_i_2_ma_1.fit(start_ar_lags = 5)
results_ar_1_i_2_ma_1.summary()


# In[420]:


df['res_ar_1_i_2_ma_1'] = results_ar_1_i_2_ma_1.resid


# In[422]:


sgt.plot_acf(df.res_ar_1_i_2_ma_1[2:], lags =40, zero = False)
plt.title("ACF for Residuals of ARIMA(1,2,1)")
plt.show()


# In[439]:


model_ar_1_i_1_ma_1_Xmarket_value = ARIMA(df.market_value, exog = df.market_value, order = (1,1,1))
results_ar_1_i_1_ma_1_Xmarket_value = model_ar_1_i_1_ma_1_Xmarket_value.fit(start_ar_lags = 10)
results_ar_1_i_1_ma_1_Xmarket_value.summary()


# In[441]:


model_sarimax = SARIMAX(df.market_value, exog = df.market_value, order = (1,0,1), seasonal_order = (2,0,1,5) )
results_sarimax = model_sarimax.fit()
results_sarimax.summary()


# In[445]:


df['returns'] = df.market_value.pct_change(1)*100


# In[447]:


df['sq_returns'] = df.returns.mul(df.returns)


# In[450]:


df.returns.plot(figsize= (20,5))
plt.title("Returns" , size = 24)
plt.show()


# In[453]:


df.sq_returns.plot(figsize= (20,5))
plt.title("Voltality" , size = 24)
plt.show()


# In[456]:


sgt.plot_pacf(df.returns[1:], lags = 40, zero = False, method = 'ols', alpha = 0.05)
plt.title("PACF of Returns", size = 24)
plt.show()


# In[458]:


sgt.plot_pacf(df.sq_returns[1:], lags = 40, zero = False, method = 'ols', alpha = 0.05)
plt.title("PACF of Voltality", size = 24)
plt.show()


# In[463]:


model_arch_1 = arch_model(df.returns[1:])
results_model_arch_1 = model_arch_1.fit(update_freq = 5)
results_model_arch_1.summary()


# In[465]:


model_arch_1 = arch_model(df.returns[1:], mean = 'Constant', vol = "Arch", p =1)
results_model_arch_1 = model_arch_1.fit(update_freq = 5)
results_model_arch_1.summary()


# In[467]:


model_arch_1 = arch_model(df.returns[1:], mean = 'AR', vol = "Arch", p =1, dist = "ged", lags = [2,3,6])
results_model_arch_1 = model_arch_1.fit(update_freq = 5)
results_model_arch_1.summary()


# In[469]:


model_arch_2 = arch_model(df.returns[1:], mean = 'Constant', vol = "Arch", p =2)
results_model_arch_2 = model_arch_2.fit(update_freq = 5)
results_model_arch_2.summary()


# In[471]:


model_arch_3 = arch_model(df.returns[1:], mean = 'Constant', vol = "Arch", p =3)
results_model_arch_3 = model_arch_3.fit(update_freq = 5)
results_model_arch_3.summary()


# In[473]:


model_arch_13 = arch_model(df.returns[1:], mean = 'Constant', vol = "Arch", p =13)
results_model_arch_13 = model_arch_13.fit(update_freq = 5)
results_model_arch_13.summary()


# In[476]:


model_garch_1_1 = arch_model(df.returns[1:], mean = 'Constant', vol = "Garch", p =1, q =1)
results_model_garch_1_1 = model_garch_1_1.fit(update_freq = 5)
results_model_garch_1_1.summary()


# In[478]:


model_garch_1_2 = arch_model(df.returns[1:], mean = 'Constant', vol = "Garch", p =1, q =2)
results_model_garch_1_2 = model_garch_1_2.fit(update_freq = 5)
results_model_garch_1_2.summary()


# In[480]:


model_garch_1_3 = arch_model(df.returns[1:], mean = 'Constant', vol = "Garch", p =1, q =3)
results_model_garch_1_3 = model_garch_1_3.fit(update_freq = 5)
results_model_garch_1_3.summary()


# In[482]:


model_garch_2_1 = arch_model(df.returns[1:], mean = 'Constant', vol = "Garch", p =2, q =1)
results_model_garch_2_1 = model_garch_2_1.fit(update_freq = 5)
results_model_garch_2_1.summary()


# In[494]:


model_garch_3_1 = arch_model(df.returns[1:], mean = 'Constant', vol = "Garch", p =3, q =1)
results_model_garch_3_1 = model_garch_3_1.fit(update_freq = 5)
results_model_garch_3_1.summary()

