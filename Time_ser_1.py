import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler

da = pd.read_csv('Airtel_6_Mar_2016_to_6_Mar_2019.csv')
dr = pd.read_csv('Reliance_6_Mar_2016_to_6_Mar_2019.csv')
dv = pd.read_csv('Vodafone_6_Mar_2016_to_6_Mar_2019.csv')
g_dr = dr['Close Price']/dr['Open Price']
g_da = da['Close Price']/da['Open Price']
g_dv = dv['Close Price']/dv['Open Price']
growth = np.column_stack((g_dr,g_da,g_dv))
train = growth[0:100]
valid = growth[100:]

print train.shape, valid.shape

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_r = scaler.fit_transform(g_dr.values.reshape(-1,1))
scaled_data_a = scaler.fit_transform(g_da.values.reshape(-1,1))
scaled_data_v = scaler.fit_transform(g_dv.values.reshape(-1,1))
#print growth[:,0].values.reshape(-1,1).shape
x_train_r, y_train_r = [], []
x_train_a, y_train_a = [], []
x_train_v, y_train_v = [], []
for i in range(20,len(train)):
    x_train_r.append(scaled_data_r[i-20:i])
    y_train_r.append(scaled_data_r[i])
    x_train_a.append(scaled_data_a[i-20:i])
    y_train_a.append(scaled_data_a[i])
    x_train_v.append(scaled_data_v[i-20:i])
    y_train_v.append(scaled_data_v[i])	
x_train_r, y_train_r = np.array(x_train_r), np.array(y_train_r)
x_train_a, y_train_a = np.array(x_train_a), np.array(y_train_a)
x_train_v, y_train_v = np.array(x_train_v), np.array(y_train_v)
x_train_r = np.reshape(x_train_r, (x_train_r.shape[0],x_train_r.shape[1],1))
x_train_a = np.reshape(x_train_a, (x_train_a.shape[0],x_train_a.shape[1],1))
x_train_v = np.reshape(x_train_v, (x_train_v.shape[0],x_train_v.shape[1],1))
print x_train_r.shape

# create and fit the LSTM network
model = Sequential()
model1 = Sequential()
model2 = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_r.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model1.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_a.shape[1],1)))
model1.add(LSTM(units=50))
model1.add(Dense(1))
model2.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_v.shape[1],1)))
model2.add(LSTM(units=50))
model2.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train_r, y_train_r, epochs=1, batch_size=1, verbose=2)
model1.compile(loss='mean_squared_error', optimizer='adam')
model1.fit(x_train_a, y_train_a, epochs=1, batch_size=1, verbose=2)
model2.compile(loss='mean_squared_error', optimizer='adam')
model2.fit(x_train_v, y_train_v, epochs=1, batch_size=1, verbose=2)
#predicting 246 values, using past 20 from the train data
inputs_r = g_dr[len(g_dr) - len(valid) - 20:].values
inputs_r = inputs_r.reshape(-1,1)
inputs_r  = scaler.transform(inputs_r)
inputs_a = g_da[len(g_da) - len(valid) - 20:].values
inputs_a = inputs_a.reshape(-1,1)
inputs_a  = scaler.transform(inputs_a)
inputs_v = g_dv[len(g_dv) - len(valid) - 20:].values
inputs_v = inputs_v.reshape(-1,1)
inputs_v  = scaler.transform(inputs_v)
X_test_r = []
X_test_a = []
X_test_v = []
for i in range(20,inputs_r.shape[0]):
    X_test_r.append(inputs_r[i-20:i,0])
    X_test_a.append(inputs_a[i-20:i,0])
    X_test_v.append(inputs_v[i-20:i,0])		
X_test_r = np.array(X_test_r)
X_test_r = np.reshape(X_test_r, (X_test_r.shape[0],X_test_r.shape[1],1))
growth_r = model.predict(X_test_r)
growth_r = scaler.inverse_transform(growth_r)
X_test_a = np.array(X_test_a)
X_test_a = np.reshape(X_test_a, (X_test_a.shape[0],X_test_a.shape[1],1))
growth_a = model.predict(X_test_a)
growth_a = scaler.inverse_transform(growth_a)
X_test_v = np.array(X_test_v)
X_test_v = np.reshape(X_test_v, (X_test_v.shape[0],X_test_v.shape[1],1))
growth_v = model.predict(X_test_v)
growth_v = scaler.inverse_transform(growth_v)
state_a = np.multiply(g_da>g_dr,g_da>g_dv)
state_r = np.multiply(g_dr>g_da,g_dr>g_dv)
state_v = np.multiply(g_dv>g_dr,g_dv>g_da)
# dictionary for encoding and decoding seq.
table = {0:'A', 1:'R',2:'V'}
# initializing data set, one hot encoded seq.
state = 1*np.column_stack((state_a,state_r,state_v))
# Array storing decoded seq.
winner = [table[np.argmax(entry)] for entry in state]
predicted_winner = [table[np.argmax(entry)] for entry in 1*np.column_stack((growth_r,growth_a,growth_v))]
print (winner)
print (predicted_winner)
plt.show()
