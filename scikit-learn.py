#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install -U scikit-learn


# In[2]:


import sklearn


# In[3]:


sklearn.__version__


# In[4]:


from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
import pandas as pd


# In[5]:


X, y = load_boston(return_X_y=True)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=1))
])
# pipe.get_params()


# In[6]:


mod = GridSearchCV(estimator=pipe,
                   param_grid={'model__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                   cv = 3)


# In[7]:


mod.fit(X, y);
pd.DataFrame(mod.cv_results_)


# In[8]:


print(load_boston()['DESCR'])


# In[ ]:




