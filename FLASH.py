#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 07:42:56 2020

@author: jonathandonesky
"""
import numpy as np 
import pandas as pd 
pd.options.display.max_columns=None
from convert_seq import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import os 


#########
# =============================================================================
# SETTING UP 
# ----------
# =============================================================================

# Switch Your Working Directory
# =============================================================================
                        # input your path
os.chdir('/Users/jonathandonesky/Desktop/FLASH/Data/15min')
#os.chdir('/Users/jonathandonesky/Desktop/FLASH/Data/1hour')


# Import Your Data
# =============================================================================
                # input your filename
data = pd.read_csv('15min_2016.csv', index_col=0)
#data = pd.read_csv('15min_full.csv', index_col=0)
#data = pd.read_csv('tutorial_data.csv', index_col=0)

# create datetime index, used for x axis when plotting results
data.index = pd.to_datetime(data.index)
time = [(str(timestamp.year)+'/'+str(timestamp.month)+'/'+str(timestamp.day)) for timestamp in data.index]


# Rearrange Columns to Put Your Target First
# =============================================================================
#####
# Below, Column Names for 1hour Interval Datasets   
# ---------------------------------------------           
#data = data[['discharge_cfs_catonsville', 'precip_onerain','precip_nldas', 'pressure',
#             'conv_pot_energy', 'discharge_cfs_hollofield']]

#####
# Below, Column Headings For Either 15min Interval Dataset
# --------------------------------------------------
data = data[['stage_ft','discharge_cfs_hollofield','discharge_cfs_catonsville','soil_moisture_kg_m2','precip_in_onerain','dry_time_h',
             'precip_in_onerain_1_h','precip_kg_m2_nldas']]  
           




# =============================================================================
# THE MODEL FUNCTION 
# ------------------


# The model is nested in a function that takes the following arguments: 
# 
# 	- data -> Pandas Dataframe 
# 
# 	- read_cols -> list of column from your data you want as inputs (with target at the front)
# 
# 	- lag_step -> number of time steps in the past (these will be the inputs to your model)
# 
# 	- forecast_step -> number of time steps in the future to forecast (this will be your predicted variable)
# 
# 	- train_split -> number of time intervals (e.g. 1hour, 15min, etc.) used for training
# 
# 	- hidden_nodes -> number of hidden nodes in LSTM layer
# 
# 	- n_epochs -> number of epochs for training (model may stop before this number is reached)
# 
# 	- batch_size -> batch size
# 
# 	- graph_file -> file path to send graphs of results  
# 
# 	- csv_file -> file path to send csv of results
#           
#       
# =============================================================================

def FLASH(data,read_cols,lag_step,forecast_step,train_split,hidden_nodes, n_epochs, batch_size,graph_file,csv_file):
   

# Pre-Processing Your Data 
# =============================================================================
    
    # Read In Your Data
    data = data[read_cols]
    num_inputs = len(read_cols)
    
    # Convert to Numpy Array of Float Values
    values = data.values
    values = values.astype('float32')
    
    # Normalize All Values to Range Between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    # Input Sliding Window Dimensions
    nlag_hours = lag_step
    n_features = num_inputs
    
    # Reframe Time Series As Sliding Window
    reframed = timeser_convert(scaled, nlag_hours, forecast_step)   
    
    # Divide Reframed Data Into Train and Test Sets 
    values = reframed.values
    n_train_hours = train_split
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    
    # Cut the Datetime Index to Fit the Reframed Window, Test Set Only 
    # (Used to timestamp your predictions in the CSV file output)
    head = reframed.index[train_split + (forecast_step-1)]
    tail = reframed.index[-1]
    output_index = data.index[head:tail+forecast_step]  
    
    # Divide Train and Test Sets into Inputs (Xs) and Outputs (y)
    # Inputs = All variables at past (lag) timesteps and present step 
    # Outputs = Target variable at user-specified forecast step (future)
    n_obs = nlag_hours * n_features + n_features  
    
    train_X, train_y = train[:, :n_obs], train[:, -n_features]  
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    
    
    
    ## Uncomment Below to Remove Your Target as an Input to Train and Test Sets
    #############
    #train_X = np.delete(train_X,list(range(0,train_X.shape[1],n_features)),axis=1) 
    #test_X = np.delete(test_X,list(range(0,test_X.shape[1],n_features)),axis=1)    
    
    
    
    # Reshape Your Inputs(Xs) to 3 Dimensions >>> [samples, timesteps, features]
    ## NOTE if you removed your target as an input, change 'n_features' to 'n_features -1' for both train_x and test_x reshapes
    train_X = train_X.reshape((train_X.shape[0], -1, n_features))# < < < < < ^^^^^^^^^^^^   
    test_X = test_X.reshape((test_X.shape[0], -1, n_features))       
     
    
    
    

# The Model 
# =============================================================================
   
    model = Sequential()                
    model.add(LSTM(hidden_nodes, input_shape=(train_X.shape[1], train_X.shape[2])))
    
    # Add or Remove Dropout Layer for performance
    # model.add(Dropout(0.2))  
    
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam',metrics=['mse','mae'])
    
    # fit network
    history = model.fit(train_X, train_y, epochs = n_epochs, batch_size=batch_size, validation_data=(test_X, test_y), 
                        callbacks = [EarlyStopping(monitor='val_loss',patience=10)],verbose=2, shuffle=False)
                                   # EarlyStopping breaks out when val_loss doesn't improve in ten epochs




# =============================================================================
# MAKING PREDICTIONS AND PLOTTING RESULTS                             
# =============================================================================
                                   
                                  
    # Create Shared Grid for All Plots
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.2,wspace=0.2)
    fig.tight_layout()
    gs = fig.add_gridspec(50,10)
    
    
    
# Plot Loss Over n Epochs
# =============================================================================
    
    ax1 = fig.add_subplot(gs[30:,:4])
    
    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Test')
    
    

# Make A Prediction
# =============================================================================
    
    # Feed Test Values To Trained Model
    y_predict = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], -1))  
    
    # Invert Scaling For Predicted Values
    offset = -(n_features - 1) 
    inv_ypredict = np.concatenate((y_predict, test_X[:, offset:]), axis=1)   
    inv_ypredict = scaler.inverse_transform(inv_ypredict)
    inv_ypredict = inv_ypredict[:,0]
    
    # Invert Scaling For Actual Values
    test_y = test_y.reshape((len(test_y), 1))                           
    inv_yactual = np.concatenate((test_y, test_X[:, offset:]), axis=1)
    inv_yactual = scaler.inverse_transform(inv_yactual)
    inv_yactual = inv_yactual[:,0]
    
    # Write Inputs And Predictions To Dataframe 
    output_df = pd.DataFrame({'Observed':inv_yactual,'Predicted':inv_ypredict},index=output_index)
    
    
    
# Assign Variables For Prediction Plot Annotations 
# =============================================================================
    
    # first test flood
    
    # actual
    actual_ser = pd.Series(inv_yactual)
    actualmax_x = actual_ser.idxmax()
    actualmax_y = actual_ser[actualmax_x]
    actualmax_y = np.floor(actualmax_y)
    
    # prediction
    pred_ser = pd.Series(inv_ypredict)
    pred_ser = pred_ser[actualmax_x-20:actualmax_x+20]
    predmax_x = pred_ser.idxmax()
    predmax_y = pred_ser[predmax_x]
    predmax_y = np.floor(predmax_y)
    
    
    difference = actualmax_y - predmax_y
    
##################################
## Uncomment If Using Ellicott Disasters I Data or Another Dataset With 
#  At Least Two Significant Flood Events in the Test Set
# =============================================================================
#     # second test flood
#     
#     # actual
#     actualmax_x = actualmax_x + 100
#     after_ser = actual_ser[actualmax_x:]
#     secnd_actualmax_x = after_ser.idxmax()
#     secnd_actualmax_y = actual_ser[secnd_actualmax_x]
#     secnd_actualmax_y = np.floor(secnd_actualmax_y)
#     
#     # prediction
#     pred_ser = pd.Series(inv_ypredict)
#     secnd_pred_ser = pred_ser[actualmax_x:]
#     
#     secnd_predmax_x = secnd_pred_ser.idxmax()
#     secnd_predmax_y = secnd_pred_ser[secnd_predmax_x]
#     secnd_predmax_y = np.floor(secnd_predmax_y)
# 
#     
#     secnd_difference = secnd_actualmax_y - secnd_predmax_y
#     
# =============================================================================
   
  
    
# Calculate Nash-Sutcliffe Efficiency
# =============================================================================
    
    def calc_nse(predicted, actual):
        nse = 1 - (sum((predicted - actual)**2)/
        sum((predicted - np.mean(actual))**2))
        return nse
    
    
    # Calculate for Test Set Prediction Only 
    nse = calc_nse(inv_ypredict,inv_yactual)
    print(f"Nash-Sutcliffe Efficiency: {nse}")


 
    
# Create Plot of Complete Dataset, Observed And Predicted
# =============================================================================
 
    # Predict Values for Train Set with Trained Model
    train_predict = model.predict(train_X,verbose=1)
    train_X = train_X.reshape((train_X.shape[0], -1))  
    
    # Invert Normalization of Result
    inv_trainpredict = np.concatenate((train_predict, train_X[:, offset:]), axis=1) 
    inv_trainpredict = scaler.inverse_transform(inv_trainpredict)
    inv_trainpredict = inv_trainpredict[:,0]
    
    # Reconcatenate Result with Test Set Predictions 
    predicted = np.concatenate((inv_trainpredict,inv_ypredict),0)

    # Take Observed Values for Complete Dataset
    actual = data['stage_ft']                    # <<<< MAKE SURE THIS COLUMN NAME MATCHES YOUR TARGET VARIABLE
    actual = actual.values
    
    ax2 = fig.add_subplot(gs[:30,:])
   
    # Plot Observed and Predicted Values for Complete Dataset
    ax2.plot(predicted,'#121ec7',linewidth=.7,label='predicted')
    ax2.plot(actual,'#1fd143',linewidth=2.5,alpha=0.3,label='actual')
             
    # Plot Divide Between Training and Test Set Predictions  
    ax2.axvline(n_train_hours,linewidth=1,color='#bd1c22',linestyle='dashed',label='train/test split')
                
    ax2.set_xticks([])     
                
## Uncomment Below If Predicting Stream Discharge 
# =============================================================================
#     thresholds= [13900.00, 20133.33, 25200.00]   
#     ax2.axhline(thresholds[0],c='#e8cf10',ls='--',alpha=0.9,linewidth=.5)
#     ax2.text(0,thresholds[0]*1.01,'Action',fontsize = 7)
#     ax2.axhline(thresholds[1],c='#e8a010',ls='--',alpha=0.9,linewidth=.5)
#     ax2.text(0,thresholds[1]*1.01,'Flood',fontsize = 7)
#     ax2.axhline(thresholds[2],c='#d12608',ls='--',alpha=0.5,linewidth=.5)
#     ax2.text(0,thresholds[2]*1.01,'Moderate Flood',fontsize = 7)
# =============================================================================



# Create Plot of Test Set Observations and Predictions
# =============================================================================
    
    ax3 = fig.add_subplot(gs[30:,4:])
    
    ax3.plot(inv_ypredict,'#121ec7',linewidth=.7,label='predicted')
    ax3.plot(inv_yactual,'#1fd143',linewidth=2.5, alpha=0.3,label='actual')
             
 ## Uncomment Below If Predicting Stream Discharge 
# =============================================================================
#     ax3.axhline(thresholds[0],c='k',ls='--',alpha=0.5,linewidth=.5)
#     #ax3.text(0,thresholds[0]*1.01,'Action',fontsize = 7)
#     ax3.axhline(thresholds[1],c='#e8a010',ls='--',alpha=0.9,linewidth=.5)
#     #ax3.text(0,thresholds[1]*1.01,'Flood',fontsize = 7)
#     ax3.axhline(thresholds[2],c='#d12608',ls='--',alpha=0.5,linewidth=.5)
#     #ax3.text(0,thresholds[2]*1.01,'Moderate Flood',fontsize = 7)
# =============================================================================
 
    ax3.vlines(actualmax_x-300,predmax_y,actualmax_y,'r',linewidth=1,linestyle='--')

# See Note on Second Test Flood Above
# =============================================================================
#     ax3.vlines(secnd_actualmax_x-175,secnd_predmax_y,secnd_actualmax_y,'r',linewidth=1,linestyle='--')
# =============================================================================

    ax3.annotate(f'{difference}', xy=(actualmax_x,predmax_y+(difference/2)),  xycoords='data',
                 xytext=(-30,0), textcoords='offset points',
                 size=6, ha='right', va="center", color = 'red',
                 bbox=dict(boxstyle="round", alpha=0.1,facecolor ='red',edgecolor='red'),
                 arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1))

# See Note on Second Test Flood    
# =============================================================================
#     ax3.annotate(f'{secnd_difference}', xy=(secnd_actualmax_x,secnd_predmax_y+(secnd_difference/2)),  xycoords='data',
#                  xytext=(-30,0), textcoords='offset points',
#                  size=6, ha='right', va="center", color = 'red',
#                  bbox=dict(boxstyle="round", alpha=0.1,facecolor ='red',edgecolor='red'),
#                  arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1))
# 
# =============================================================================
    
    ax3.set_yticks([])
    ax3.set_xticks([])
# =============================================================================
#     #ax3_xlabels = ax2.get_xticklabels()
#     #ax3.set_xticklabels(ax3_xlabels,rotation=40,ha='right')
# =============================================================================
    
    # For 15min interval data 
    txt = f" Inputs: {read_cols} \n Lag: {lag_step/4} hour(s) \n Forecast: {(forecast_step-1)/4} hour(s)  \n Train Set: {n_train_hours/4} hours \n Test Set: {len(test)/4} hours \n Nodes: {hidden_nodes} \n Batch Size: {batch_size} \n NSE: {nse}"
    # For H interval data
    #txt = f" Inputs: {read_cols} \n Lag: {lag_step} hour(s) \n Forecast: {forecast_step-1} hour(s)  \n Train Set: {n_train_hours} hours \n Test Set: {len(test)} hours \n Nodes: {hidden_nodes} \n Batch Size: {batch_size} \n NSE: {nse}"
    plt.figtext(0.93,0.2, txt, wrap=True,va='bottom', ha ='left', fontsize=6)
    


# Send Plotted and Tabulated Results to Graph and CSV Files For Sensitivity Analysis (Below)
# =============================================================================
    fig.savefig(graph_file,dpi=500,bbox_inches='tight')
    output_df.to_csv(csv_file)
    
    
    
    # For Nash-Sutcliffe Efficiency Loop 
    return nse


  
# # Option to Uncomment Plot 1 or 2 Below When Using Ellicott Disaster I Data (Target = Discharge)
# =============================================================================     
## If You Do, Move 'return nse' Below the Block

# Plot 1
# Precipitation Coming Down From a Twinned Secondary Axis 
#####################
  
#     fig, plt = plt.subplots()
#     ax2 = plt.twinx() 
#     
#     plt.plot(predicted,'#121ec7',linewidth=.7,label='predicted')
#     plt.plot(actual,'#1fd143',linewidth=2.5,alpha=0.3,label='actual')
#     plt.axvline(n_train_hours,linewidth=1,color='#bd1c22',linestyle='dashed',label='train/test split')
#     plt.set_title(f'Train & Test Sets: Discharge & Precipitation')
#     plt.set_ylabel('Discharge (cfs)')
#     plt.set_xlabel('\n Time (hrs)')
#     
#     @ticker.FuncFormatter                
#     def fmtr(x, pos):
#         label = str(-x) if x < 0 else str(x)
#         return label
#     
#     ax2.plot(-precip,'#03a9fc',linewidth=3.5,alpha=0.2,label='Precipitation')              
#     ax2.set_ylabel('\n Precipitation (kg/m2)',va='top')
#     ax2.yaxis.set_major_formatter(fmtr)
#     
#     plt.legend(loc='upper right',fancybox=True,framealpha=.5,borderpad=.5)
#     
#     plt.xticks(np.arange(0,len(time)+1,5000),time[::5000])
#     
  

# Plot 2 
# With Precipitation on Twinned Secondary Axis And  moisture data on secondary axis below discharge_ax
#########################3
#     fig = plt.figure(figsize=(6,6))
#     grid = plt.GridSpec(4,8,hspace=.1)
#     
#     discharge_ax = fig.add_subplot(grid[:-2,:])
#     precip_ax = discharge_ax.twinx() 
#     soil_ax = fig.add_subplot(grid[-2,:])
#     
#     discharge_ax.plot(predicted,'#121ec7',linewidth=.7,label='predicted')
#     discharge_ax.plot(actual,'#1fd143',linewidth=2.5,alpha=0.3,label='actual')
#     discharge_ax.axvline(n_train_hours,linewidth=1,color='#bd1c22',linestyle='dashed',label='train/test split')
#     discharge_ax.set_title(f'Hourly Precip and Discharge Over Time')
#     discharge_ax.set_ylabel('Discharge (cfs)')
#     discharge_ax.legend(loc='upper right',fancybox=True,framealpha=.5,borderpad=.5)
#     
#     precip_ax.plot(-precip,'#03a9fc',linewidth=3.5,alpha=0.2,label='Precipitation')              
#     precip_ax.set_ylabel('\n Precipitation (kg/m2)',va='top')
#     precip_ax.legend(loc='upper left',fancybox=True,framealpha=.5,borderpad=.5)
#     
#     discharge_ax.xaxis.set_major_locator(plt.NullLocator())
#     discharge_ax.xaxis.set_major_formatter(plt.NullFormatter())
#     precip_ax.xaxis.set_major_locator(plt.NullLocator())
#     precip_ax.xaxis.set_major_formatter(plt.NullFormatter())
#   
#     soil_ax.plot(soil,'#ebb01c',linewidth=2.5,alpha=0.8,label='Soil Moisture')
#     soil_ax.set_xlabel('\n Time (hrs)')
#     soil_ax.set_ylabel('\n Soil Moisture (kg/m2)',va='top')
#     soil_ax.legend(loc='upper left',fancybox=True,framealpha=.5,borderpad=.5)
#     soil_ax.yaxis.set_label_position('right')
#     soil_ax.yaxis.tick_right()
#     
#     plt.xticks(rotation=90)
#     plt.xticks(np.arange(0,len(time)+1,5000),time[::5000])
#     
#     @ticker.FuncFormatter
#     def fmtr(x, pos):
#         label = str(-x) if x < 0 else str(x)
#         return label
#     precip_ax.yaxis.set_major_formatter(fmtr)






# ==========================================================================================================================================================
# =========================================================================================================================================================
    
# RUNNING FLASH 
# -------------
    
# =========================================================================================================================================================

# Fixed-Parameters
# =============================================================================

# Choose Fixed Parameters 
fixed_parameters = dict(
                  data = data,   
                  
                  read_cols=['stage_ft','discharge_cfs_hollofield','discharge_cfs_catonsville',
                             'soil_moisture_kg_m2','precip_in_onerain'],
                  
                  lag_step = 96,   # 24 if using 1 hour data        
                  
                  forecast_step = 2,   # must be > 1 
                  
                  train_split = 70080,  # 35000 - 40000 if using 1 hour data
                  
                  hidden_nodes = 50,
                  
                  n_epochs = 50,     
                  
                  batch_size = 72,
                  
                  #graph_file = graph_file,
                  #csv_file = csv_file, 
                  )


# Create Folder To Store Result
# Copy Absolute Path To That Folder Below
baseDir = '/Users/jonathandonesky/Desktop/single_result'

# Run Model Min 3 times With Fixed Parameters
for run in range(1,4): 
    graph_file = open(baseDir + f'/GRAPH_run{run}.png','wb')
    csv_file = open(baseDir + f'/CSV_run{run}.csv','w')
    FLASH(**fixed_parameters,graph_file=graph_file,csv_file=csv_file)
    plt.close()  


# =============================================================================
# Sensitivity Analysis
# --------------------
# Run the model in a loop, outputting results for a range of parameters and/or combinations of inputs 
    
# STEP 1 - 
# Choose Which Parameters To Loop For Analysis And Which To Keep Fixed 
# =============================================================================

loop_parameters = dict(
        
                  data = data,   
                  
                  read_cols=['stage_ft','discharge_cfs_hollofield','discharge_cfs_catonsville',
                             'soil_moisture_kg_m2','precip_in_onerain','dry_time_h',
                             'precip_in_onerain_1_h','precip_kg_m2_nldas'],
                  
                  lag_step = 96,   # 24 if using 1 hour data        
                  
                  #forecast_step = LOOP RANGE OF VALUES   
                  
                  train_split = 70080,  # 35000 - 40000 if using 1 hour data
                  
                  hidden_nodes = 50,
                  
                  n_epochs = 50,     
                  
                  batch_size = 72,
                  
                  #graph_file = graph_file,
                  #csv_file = csv_file,
                  )



# STEP 2 -
# Create Base Directory to Store Results 
# =============================================================================
baseDir = '/Users/jonathandonesky/Desktop/result_loop'
#baseDir = 'C:/Users/jdonesky/Desktop/results_loop'



# STEP 3 -
# Select Loop
# =============================================================================

## LOOP FORECASTS STEPS ONLY
#  -----------------------------
# =============================================================================
# #  'step' Refers to Forecast_step parameter -> NOTE step = 2 means model is predicting 1 timestep ahead (e.g. 1hr,15min,etc)
# for step in range(2,13):        
#     dirname = 'forecast' + str((step-1)/4) + 'hours'  # (forecast-1)/4 for 15 min interval -> hours
#     subdir= os.path.join(baseDir,dirname)
#     os.mkdir(subdir)
#     for run in range(1,4): 
#         graph_file = open(subdir+ f'/GRAPH_{dirname}_run{run}.png','wb')
#         csv_file = open(subdir + f'/CSV_{dirname}_run{run}.csv','w')
#         FLASH(**loop_parameters,forecast_step=step,graph_file=graph_file,csv_file=csv_file)
#         plt.close()    
# 
# =============================================================================


## LOOP BOTH INPUTS AND FORECAST STEPS
#  --------------------------------------
# =============================================================================
# 
# read_cols=['stage_ft','discharge_cfs_hollofield','discharge_cfs_catonsville',
#                              'soil_moisture_kg_m2','precip_in_onerain','dry_time_h',
#                               'precip_in_onerain_1_h','precip_kg_m2_nldas']
# 
# for i in range(1,len(read_cols)+1):
#      cols = read_cols[:i]
#      for step in range(2,13):  # (2,14)
#          dirname = str(cols)+'_'+ 'forecast ' + str(step)
#          subdir= os.path.join(baseDir,dirname)
#          os.mkdir(subdir)
#          for run in range(1,4): # (1,4)
#              graph_file = open(subdir+ f'/GRAPH_{dirname}_run{run}.png','wb')
#              csv_file = open(subdir + f'/CSV_{dirname}_run{run}.csv','w')
#              FLASH(**loop_parameters,read_cols=cols,forecast_step=step,graph_file=graph_file,csv_file=csv_file)
#              plt.close()
# 
# 
# =============================================================================
## LOOP NASH-SUTCLIFFE EFFICIENCY ONLY
#  --------------------------------------        
# Remove graph and csv file parameters to produce only nse output for faster loop
        
        
# fig, ax = plt.subplots(figsize=(10,10))
    
# labels, nse_vals = [],[] 

# i = 1
# for lag_step in range(1,24):
#     for forecast_step in range(1,24):
#         nse = FLASH(**parameters,lag_step=lag_step,forecast_step=forecast_step)
#         label = (lag_step,forecast_step)
#         labels.append(label)
#         nse_vals.append(nse)
#         ax.plot(i,nse,'bo',markersize=12,alpha=0.5)
#         plt.text(i*(1.01),nse*(1.01),str(label),fontsize=7)
#         i += 1 

# labeled = zip(labels,nse_vals)
# print(labeled)

# for item in labeled: 
#     print(f' lag/forecast:{item[0]} \n nse:{item[1]}')


# plt.grid(True,'major','y',linestyle='--')
# #ax.set_xticklabels([str(label) for label in labels])
# ax.xaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())
# ax.set_xlabel('\n (Lag,Forecast)')
# ax.set_ylabel('Nash Sutcliffe Efficiency \n')
# ax.set_ylim(-.1,1.1)
# plt.show()
        
        
        