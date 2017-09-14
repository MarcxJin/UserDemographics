


# In[1]:

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

import os
from scipy.sparse import csr_matrix, hstack


# In[2]:

seed = 7
np.random.seed(seed)

# load dataset
#dataframe = pandas.read_csv("/home/username/projects/PracticeNN/iris.data", header=None)

datadir = '/Users/Derick/Desktop/talkingdata'
#datadir = '/home/username/projects/talkingData/input'
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'), index_col='device_id')
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'), index_col = 'device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv(os.path.join(datadir,'events.csv'),  parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), usecols=['event_id','app_id','is_active'], dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))


# In[3]:

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


# In[4]:

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


# In[5]:

appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['size'])
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())
deviceapps.head()

d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)),
                      shape=(gatrain.shape[0],napps))
d = deviceapps.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)),
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))


# In[6]:

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

d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)),
                      shape=(gatrain.shape[0],nlabels))
d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)),
                      shape=(gatest.shape[0],nlabels))
print('Labels data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))


# In[14]:

# active_time
a = pd.DataFrame(events['timestamp'],index = (events['timestamp']))
hour = a.index.hour
weekday = a.index.weekday
events['timeitv'] = (weekday // np.linspace(7,7,events.shape[0])) * 7 + (hour // np.linspace(3,3,events.shape[0]))
timecoder = LabelEncoder().fit(events.timeitv)
#events['timeitv'] = timecoder.transform(events.timeitv)
ntime = len(timecoder.classes_)
time = events[['device_id','timeitv']].groupby(['device_id','timeitv'])['timeitv'].agg(['size']).merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True).merge(gatest[['testrow']], how='left', left_index=True, right_index=True).reset_index()
time.head()

d = time.dropna(subset=['trainrow'])
Xtr_time = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.timeitv)), 
                      shape=(gatrain.shape[0],ntime))
d = time.dropna(subset=['testrow'])
Xte_time = csr_matrix((np.ones(d.shape[0]) , (d.testrow, d.timeitv)), 
                      shape=(gatest.shape[0],ntime))
print('time data: train shape {}, test shape {}'.format(Xtr_time.shape, Xte_time.shape))


# In[15]:

zipcode=pd.read_csv('/Users/Derick/Desktop/talkingdata/zipcode.csv')
nzips=531
d = zipcode.dropna(subset=['trainrow'])
Xtr_location = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d['zip'])), 
                      shape=(gatrain.shape[0],nzips))
d = zipcode.dropna(subset=['testrow'])
Xte_location = csr_matrix((np.ones(d.shape[0]) , (d.testrow, d['zip'])), 
                      shape=(gatest.shape[0],nzips))
print('location data: train shape {}, test shape {}'.format(Xtr_location.shape, Xte_location.shape))


# In[16]:

Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label, Xtr_time, Xtr_location), format='csr')
Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label, Xte_time, Xte_location), format='csr')
print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))


# In[17]:

targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)
nclasses = len(targetencoder.classes_)
dummy_y = np_utils.to_categorical(y) ## Funcion de Keras!


# In[18]:

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


# In[28]:

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches =X.shape[0] /  np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0


# In[36]:

from keras.layers.advanced_activations import PReLU
# build model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=Xtrain.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50, input_dim=Xtrain.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model

model=baseline_model()
X_train, X_val, y_train, y_val = train_test_split(Xtrain, dummy_y, test_size=0.002, random_state=42)
fit= model.fit_generator(generator=batch_generator(X_train, y_train, 400, True),
                         nb_epoch=15,
                         samples_per_epoch=69984,
                         validation_data=(X_val.todense(), y_val), verbose=2
                         )


# In[29]:




# In[38]:

# evaluate the model
scores_val = model.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
print('logloss val {}'.format(log_loss(y_val, scores_val)))

print("# Final prediction")
scores = model.predict_generator(generator=batch_generatorp(Xtest, 800, False), val_samples=Xtest.shape[0])
pred = pd.DataFrame(scores, index = gatest.index, columns=targetencoder.classes_)
pred.to_csv('submit16.csv',index=True)






# In[ ]:





