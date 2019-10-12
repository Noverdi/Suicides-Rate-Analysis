#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ###  exploring data

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('./doc/master.csv')


# In[2]:


list(df)


# In[3]:


# Formatting columns name
df.rename(columns={'suicides/100k pop':'suicides/100k_pop',
                  'country-year': 'country_year',
                  'HDI for year':'HDI',
                  ' gdp_for_year ($) ':'gdp_for_year',
                  'gdp_per_capita ($)':'gdp_percapita'}, inplace=True)


# In[4]:


# data info
df.info()


# In[5]:


df['gdp_for_year'] = df.gdp_for_year.apply(lambda x:x.replace(",","")).astype(np.int64)


# In[6]:


df.sample(5)


# ### Performing Statistical Analysis
# #### GDP and Suicide
# GDP maybe a variable that may concern by people who live in that particular country. Maybe it's the one factor that cause suicide rate to increase, that why I want to perform some test to see what really going on here. I'm asking for the cause in this section, but I'm not finding the cause in this section, I'm looking for the association here.
# 
# Since I'm going to find the correlation between suicide_no and gdp_for_year. I have to check the assumption first.

# In[7]:


df.suicides_no.describe()


# In[8]:


df.suicides_no.plot(grid=True,kind='hist',figsize=(15,8))
plt.xlabel('suicides_no')
plt.ylabel('Frequency')
plt.show()


# In[9]:


# Calc Skewness
df.suicides_no.skew()


# In order to perform test properly. Skewness is so high (10.35179). I have to pick either to transform this variable and pick Pearson as a method to perform correlation test. Because in order to use Pearson as a method, your variable should be normally distributed. The easy solution how to tell your variable is normally distributed or not is to look at Skewness value. Skewness value should not larger than one (Skewness < 1) . Or alternative solution, I don't have to transform this variable, but I have to pick Kendall or Spearman as a method to perform correlation test. Let look at gdp_for_year variable and find its skewness.

# In[10]:


df.gdp_for_year.describe()


# In[11]:


df.gdp_for_year.plot(grid=True,kind='hist',figsize=(15,8))
plt.xlabel('gdp_for_year')
plt.ylabel('Frequency')
plt.show()


# In[12]:


df.gdp_for_year.skew()


# Since both variables are not normally distributed (Their skewness are higher than 1). Therefore I will use Spearman as a method to find the correlation between those two variable.

# In[13]:


df['suicides_no'].corr(df['gdp_for_year'],method='spearman')


# As a result appear, There is significantly correlated between suicides_no and gdp_for_year. But this correlation is not that strong. Correlation matrix is only 0.6586 (0.8 is considering a strong relationship). Since what we have here is only a sample. Let look at the confident interval and see there is any correlation in population or not.

# In[18]:


r_z = np.arctanh(df['suicides_no'].corr(df['gdp_for_year'],method='spearman'))
se = 1/np.sqrt(df.suicides_no.count()-3)
alpha = 0.05
z = stats.norm.ppf(1-alpha/2)
lo_z, hi_z = r_z-z*se, r_z+z*se
lo, hi = np.tanh((lo_z, hi_z))


# In[15]:


# Confidence Interval
lo,df['suicides_no'].corr(df['gdp_for_year'],method='spearman'),hi


# There is correlation between these variable in population. And it's lies between 0.6518741 and 0.6651833. Statistically speaking, this interval consider to be so small and almost certain.

# #### HDI(Human Development Index) and Suicide
# Apart from income or money, HDI variable is also the one I think it may cause the problem. Maybe the person who live in certain country where they don't see any developments in well-being of people or development in general. They may end their live. They may think "what the point to do this hard work for".
# 
# Like previous analysis. we must check the assumption and find the value of skewness in order to pick the optimum method to perform any test.

# In[16]:


#Checking the HDI assumption
df.HDI.describe(), "Null = %a" % df.HDI.isna().sum(), "skew = %s" %df.HDI.skew()


# In[17]:


df.HDI.plot(kind='hist', bins=30, grid=True, figsize=(15,8))


# Since skewness of HDI for year is less than 1 (HDI for year = 0.3). It's considered to be normally distributed variable. Not a perfectly bell curve, but it's acceptable.
# 
# Someone may think, Since this variable is normally distributed. We can choose Pearson method in correlation test. But I can't do that now. Because other variable (suicides_no) that we want to know the correlation from is not normally distributed. We have to transform suicides_no variable. before perform a test.

# In[7]:


# data description
df.describe()


# In[9]:


df['age'].unique()


# In[9]:


df['age']=df['age'].replace('5-14 years','05-14 years')


# In[10]:


df.age.unique()


# In[11]:


df['suicides_rate']=df.suicides_no/df.population


# In[23]:


year = df.groupby(by='year')[['suicides_no','population','suicides/100k pop']].sum()


# In[24]:


year['population'].plot(figsize = (18,8))


# In[19]:


df['suicides_rate'].plot(kind='hist', title= 'Suicides rate Histogram')


# In[20]:


df['suicides_no'].plot(kind='hist')


# In[39]:


df['suicides_no_log']=df['suicides_no'].apply(lambda x: math.log(x+1))


# In[57]:


df.suicides_no_log.plot(kind='hist',
                        figsize=(15,5))


# In[66]:


df['suicides_no_log'].describe()


# In[65]:


df['suicides_no_log'].skew()


# In[25]:


df['HDI for year'].plot(kind='hist',figsize=(15,5))


# In[67]:


df.corr()


# In[49]:


b.nlargest(10,'suicides_rate')


# In[ ]:





# In[6]:


a['age'] = a['age'].replace({'5-14 years':'05-14 years'})


# In[7]:


age = pd.pivot_table(a,index = 'age')


# In[ ]:





# In[56]:


age.suicides_no.describe()


# In[26]:


age['suicide_rate']=age['suicides_no']/age['population']*100


# In[29]:


country = pd.pivot_table(a,index = 'country')


# In[81]:


a = list(country.nlargest(10,'suicides_no').index)
b = list(country.nlargest(10,'suicides/100k pop').index)
z = zip(a,b)
set(z)


# In[58]:


country['suicide_rate']=country['suicides_no']/country['population']*100


# In[43]:


country.suicides_no.plot(kind='bar',figsize = (18,8)) # figsize : a tuple (width, height) in inches


# In[45]:


country.population.plot(kind='bar',figsize = (18,8)) # figsize : a tuple (width, height) in inches


# In[64]:


country.suicide_rate.sort_values(ascending=False).plot(kind='bar',figsize = (18,8))


# In[57]:


a.corr()


# In[5]:


b = pd.read_html('https://en.wikipedia.org/wiki/Islam_by_country')


# In[12]:


b1 = b[3][['Country/Region','Total Population','Muslim Population','Muslim percentage (%) of total population']]


# In[18]:


# group by country and age
c_a = a.groupby(by=['country','age','year'])


# In[19]:


c_a.sum()


# In[36]:


men = a[a.sex == 'male']
women = a[a.sex == 'female']


# In[39]:


import seaborn as sns


# In[41]:


import matplotlib.pyplot as plt


# In[58]:


sns.lineplot(men.year, a.population, ci = None)
sns.lineplot(women.year, a.population, ci = None)
#sns.lineplot(a.year, a.population/10000, ci = None)
plt.legend(["male", 'female'])
plt.show()


# In[67]:


import numpy as np


# In[68]:


a[' gdp_for_year ($) '] = a[' gdp_for_year ($) '].str.replace(",","").astype(np.int64)


# In[69]:


a[' gdp_for_year ($) '].sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




