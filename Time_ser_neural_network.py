#setting up a NN for winner prediction from a competing set of three Telecommunication companies
#Using stock market data for the three companies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
da = pd.read_csv('Airtel_6_Mar_2016_to_6_Mar_2019.csv')
dr = pd.read_csv('Reliance_6_Mar_2016_to_6_Mar_2019.csv')
dv = pd.read_csv('Vodafone_6_Mar_2016_to_6_Mar_2019.csv')
print dv.shape

#g = growth ratio = (closing price)/(opening price)
g_da = da['Close Price']/da['Open Price']
g_dr = dr['Close Price']/dr['Open Price']
g_dv = dv['Close Price']/dv['Open Price']
#state determination one with the highest growth rate will be in state 1 else will be in state 0
state_a = np.multiply(g_da>g_dr,g_da>g_dv)
state_r = np.multiply(g_dr>g_da,g_dr>g_dv)
state_v = np.multiply(g_dv>g_dr,g_dv>g_da)
# dictionary for encoding and decoding seq.
table = {0:'A', 1:'R',2:'V'}
# initializing data set, one hot encoded seq.
state = 1*np.column_stack((state_a,state_r,state_v))
# Array storing decoded seq.
winner = [table[np.argmax(entry)] for entry in state]
# scatter plot with color scheme
dict_color = {'A':'Green','R':'Red','V':'Blue'}
#i=1
#for entry in w:
#	plt.scatter(i,1,color=dict_color[entry])
#	i=i+1	
#training data-set
train = state[0:50]
valid = state[50:]
# preparing data for LSTM
x_train, y_train = [], []
t=25
for i in range(t,len(train)):
    x_train.append(state[i-t:i])
    y_train.append(state[i])
x_train, y_train = np.array(x_train), np.array(y_train)
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(x_train.shape[2]))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 99 values, using past t from the train data
inputs = state[len(state) - len(valid) - t:]
x_test=[]
for i in range(t,inputs.shape[0]):
	x_test.append(inputs[i-t:i])
x_test = np.array(x_test)
winner_predict = model.predict(x_test)
print winner_predict
winner_predict_bin = np.column_stack((np.multiply(winner_predict[:,0]>winner_predict[:,1] , winner_predict[:,0]>winner_predict[:,2]),np.multiply(winner_predict[:,1]>winner_predict[:,0] , winner_predict[:,1]>winner_predict[:,2]),np.multiply(winner_predict[:,2]>winner_predict[:,0] , winner_predict[:,2]>winner_predict[:,1])))
winner_predict_char = [table[np.argmax(entry)] for entry in winner_predict_bin]
print winner_predict_bin
#i=1
#for entry in winner:
#	plt.scatter(i,1,color=dict_color[entry])
#	i=i+1	
#i=1
#for entry in winner_predict_char:
#	plt.scatter(len(train)+i-1,2,color=dict_color[entry])
#	i=i+1
#plt.show()
truth_val = 0
#print len(range(t,inputs.shape[0])),len(winner_predict_bin)
for i in range(t,len(winner_predict_bin)):
	print winner_predict_bin[i],state[i],np.array_equal(winner_predict_bin[i],state[i])
	truth_val= truth_val + np.array_equal(winner_predict_bin[i],state[i])
	print truth_val
print truth_val,len(x_test)
#print [table[np.argmax(entry)] for entry in 1*winner_predict_bin]
#print '**********************'

