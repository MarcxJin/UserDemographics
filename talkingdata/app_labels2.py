
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import matplotlib.pyplot as plt
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from geopy.geocoders import Nominatim


# In[2]:

datadir = 'Desktop/talkingdata'
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),
                      index_col='device_id')
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),
                     index_col = 'device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv(os.path.join(datadir,'events.csv'),
                     parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), 
                        usecols=['event_id','app_id','is_active'],dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))


# In[30]:

# phone brand
gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])
brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.brand)))
Xte_brand = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.brand)))
print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))


# In[43]:

# phone model
m = phone.phone_brand.str.cat(phone.device_model)
modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']
Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.model)))
Xte_model = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.model)))
print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))


# In[44]:

# app size
appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['size'])
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())
deviceapps.head()


# In[45]:

d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]) , (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = deviceapps.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]) , (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Size data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))


# In[46]:

# device labels
applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
applabels['app'] = appencoder.transform(applabels.app_id)
labelencoder = LabelEncoder().fit(applabels.label_id)
applabels['label'] = labelencoder.transform(applabels.label_id)
nlabels = len(labelencoder.classes_)
devicelabels = (deviceapps[['device_id','app']]
                .merge(applabels[['app','label']])
                .groupby(['device_id','label'])['app'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
devicelabels.head()


# In[47]:

d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]) , (d.trainrow, d.label)), 
                      shape=(gatrain.shape[0],nlabels))
d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]) , (d.testrow, d.label)), 
                      shape=(gatest.shape[0],nlabels))
print('Labels data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))


# In[48]:

# active_time
a = pd.DataFrame(events['timestamp'],index = (events['timestamp']))
hour = a.index.hour
events['timeitv'] = hour // np.linspace(3,3,events.shape[0])
timecoder = LabelEncoder().fit(events.timeitv)
#events['timeitv'] = timecoder.transform(events.timeitv)
ntime = len(timecoder.classes_)
time = events[['device_id','timeitv']].groupby(['device_id','timeitv'])['timeitv'].agg(['size']).merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True).merge(gatest[['testrow']], how='left', left_index=True, right_index=True).reset_index()
time.head()


# In[49]:

d = time.dropna(subset=['trainrow'])
Xtr_time = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.timeitv)), 
                      shape=(gatrain.shape[0],ntime))
d = time.dropna(subset=['testrow'])
Xte_time = csr_matrix((np.ones(d.shape[0]) , (d.testrow, d.timeitv)), 
                      shape=(gatest.shape[0],ntime))
print('time data: train shape {}, test shape {}'.format(Xtr_time.shape, Xte_time.shape))


# In[7]:

#appencoder = LabelEncoder().fit(appevents.app_id)
#appevents['app'] = appencoder.transform(appevents.app_id)
#napps = len(appencoder.classes_)
#deviceapps0 = appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True).groupby(['device_id','app'])['app'].agg(['size']).reset_index()


# In[8]:

#deviceapps1 = appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True).groupby(['device_id','app'])['is_active'].mean().to_frame().reset_index()
#result = pd.merge(deviceapps0, deviceapps1, on=['device_id','app'])
#deviceapps=result.merge(gatrain[['trainrow']], how='left', left_on='device_id',right_index=True).merge(gatest[['testrow']], how='left', left_on='device_id',right_index=True)
#deviceapps.head()


# In[50]:

location_hash = {}
count = 0
for lat in range(25, 61):
    for lon in range(75, 136):
        location_key = str(lat) + ', ' + str(lon)
        location_hash[location_key] = count
        count+=1
#pickle.dump(location_hash, open( "location_hash.p", "wb" ) )
print('hash table ready')


# In[51]:

foreignCount = 0
zeroCount = 0
def get_zip(row):
    global zeroCount, foreignCount
    pos = str(int(row.latitude)) + ', ' + str(int(row.longitude))
    if int(row.latitude) == 0 and int(row.longitude) == 0 or int(row.latitude) == 1 and int(row.longitude) == 1:
        zeroCount += 1
        return
    
    if not (pos in location_hash):
        foreignCount+=1
        return 0
    else:
        return location_hash[pos]


# In[52]:

events['zip'] = events.apply(get_zip, axis = 1)
print(foreignCount)
print(zeroCount)
events.head()


# In[53]:

events = events.dropna(subset=['zip'])
zipencoder = LabelEncoder().fit(events['zip'])
events['zip'] = zipencoder.transform(events['zip'])
nzips = len(zipencoder.classes_)
print(nzips)
zipcode = events[['device_id','zip']].groupby(['device_id','zip'])['zip'].agg(['size']).merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True).merge(gatest[['testrow']], how='left', left_index=True, right_index=True).reset_index()
zipcode.head(10)


# In[54]:

d = zipcode.dropna(subset=['trainrow'])
Xtr_location = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d['zip'])), 
                      shape=(gatrain.shape[0],nzips))
d = zipcode.dropna(subset=['testrow'])
Xte_location = csr_matrix((np.ones(d.shape[0]) , (d.testrow, d['zip'])), 
                      shape=(gatest.shape[0],nzips))
print('time data: train shape {}, test shape {}'.format(Xtr_location.shape, Xte_location.shape))


# In[55]:

# stack all features together
Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label, Xtr_time, Xtr_location), format='csr')
Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label, Xte_time, Xte_location), format='csr')
print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))


# In[56]:

# cross validation
targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)
nclasses = len(targetencoder.classes_)


# In[57]:

def score(clf, random_state = 0):
    kf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
    pred = np.zeros((y.shape[0],nclasses))
    for itrain, itest in kf:
        Xtr, Xte = Xtrain[itrain, :], Xtrain[itest, :]
        ytr, yte = y[itrain], y[itest]
        clf.fit(Xtr, ytr)
        pred[itest,:] = clf.predict_proba(Xte)
        # Downsize to one fold only for kernels
        return log_loss(yte, pred[itest, :])
        print("{:.5f}".format(log_loss(yte, pred[itest,:])))
    print('')
    return log_loss(y, pred)


# In[48]:

#clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(6), random_state=1)
#clf.fit(Xtrain, y)
#score(clf)


# In[62]:

Cs = np.logspace(-3,-1,10)
res = []
for C in Cs:
    res.append(score(LogisticRegression(C = C, multi_class='multinomial',solver='newton-cg')))
plt.semilogx(Cs, res,'-o');


# In[59]:

score(LogisticRegression(C=0.019, multi_class='multinomial',solver='newton-cg'))


# In[60]:

score(LogisticRegression(C=0.01, multi_class='multinomial',solver='newton-cg'))


# In[61]:

score(LogisticRegression(C=0.02, multi_class='multinomial',solver='newton-cg'))


# In[72]:

clf = LogisticRegression(C=0.0195, multi_class='multinomial',solver='newton-cg')
clf.fit(Xtrain, y)
pred = pd.DataFrame(clf.predict_proba(Xtest), index = gatest.index, columns=targetencoder.classes_)
pred.head()


# In[73]:

pred.to_csv('submit_7.csv',index=True)


# In[50]:

# xgboost
params = {}
params['booster'] = 'gblinear'
params['objective'] = "multi:softprob"
params['eval_metric'] = 'mlogloss'
params['eta'] = 0.005
params['num_class'] = 12
params['lambda'] = 3
params['alpha'] = 2

# Random 10% for validation
kf = list(StratifiedKFold(y, n_folds=10, shuffle=True, random_state=4242))[0]

Xtr, Xte = Xtrain[kf[0], :], Xtrain[kf[1], :]
ytr, yte = y[kf[0]], y[kf[1]]

print('Training set: ' + str(Xtr.shape))
print('Validation set: ' + str(Xte.shape))


# In[51]:

d_train = xgb.DMatrix(Xtr, label=ytr)
d_valid = xgb.DMatrix(Xte, label=yte)

watchlist = [(d_train, 'train'), (d_valid, 'eval')]

clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=25)

pred = clf.predict(xgb.DMatrix(Xtest))

pred = pd.DataFrame(pred, index = gatest.index, columns=targetencoder.classes_)
pred.head()


# In[ ]:

pred.to_csv('sparse_xgb.csv', index=True)

