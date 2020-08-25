import numpy as np 
import pandas as pd
pd.options.display.max_columns=None

# convert series to supervised learning
def timeser_convert(data, n_in=1, n_out=1, dropnan=True):
	num_variables = 1 if type(data) is list else data.shape[1]   # treats TIME as the variable <<< 
	df = pd.DataFrame(data)                                      
	cols, names = [],[]
	# input sequence (previous timesteps)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(num_variables)]
	# forecast sequence (current or future timesteps - forecast target)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(num_variables)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(num_variables)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


'''
Notes

function is based on > pandas shift() function 

given a dataframe, shift can create copies of columns that are pulled backward (i) 
or pushed forward (-i) up/down the index (in this case time)

this creates columns of lag observations and forecast observations


'''
#t = np.arange(0,10)
#df = pd.DataFrame({'t':t})
#df['t-1'] = df['t'].shift(1)

#
#df['t+1'] = df['t'].shift(-1)
#df.drop('t-1',1,inplace=True)
#   t  t+1
#0  0  1.0
#1  1  2.0
#2  2  3.0
#3  3  4.0
#4  4  5.0
#5  5  6.0
#6  6  7.0
#7  7  8.0
#8  8  9.0
#9  9  NaN

'''
Also works on multivariate time series problems
that is where instead of having one set of observations for a time series, we have multiple (e.g. temperature, pressure)

All variates in the time series can be shifted forward or backward to create multivariate input and output sequences

'''

#values = [x for x in range(10)]
#data = timeser_convert(values,n_in=3)
#   var1(t-3)  var1(t-2)  var1(t-1)  var1(t)
#3        0.0        1.0        2.0        3
#4        1.0        2.0        3.0        4
#5        2.0        3.0        4.0        5
#6        3.0        4.0        5.0        6
#7        4.0        5.0        6.0        7
#8        5.0        6.0        7.0        8
#9        6.0        7.0        8.0        9



#obs1 = [x for x in range(10)]
#obs2 = [x for x in range(50,60)]
#
#df = pd.DataFrame({'obs1':obs1,'obs2':obs2})
#
#data = timeser_convert(df,1,3)
#print(data)
#   var1(t-1)  var2(t-1)  var1(t)  var2(t)  var1(t+1)  var2(t+1)  var1(t+2)  \
#1        0.0       50.0        1       51        2.0       52.0        3.0   
#2        1.0       51.0        2       52        3.0       53.0        4.0   
#3        2.0       52.0        3       53        4.0       54.0        5.0   
#4        3.0       53.0        4       54        5.0       55.0        6.0   
#5        4.0       54.0        5       55        6.0       56.0        7.0   
#6        5.0       55.0        6       56        7.0       57.0        8.0   
#7        6.0       56.0        7       57        8.0       58.0        9.0   
#
#   var2(t+2)  
#1       53.0  
#2       54.0  
#3       55.0  
#4       56.0  
#5       57.0  
#6       58.0  
#7       59.0  