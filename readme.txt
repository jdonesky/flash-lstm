
---------
Contents 
---------

This file contains guidance for using:

	(1) time series conversion function

	(2) LSTM model

	(3) sensitivity analysis loop 

	(4) scripts for USGS data extraction, resample, export to excel, and merge 



-----------------
Required Packages 
-----------------

* Keras 
* Numpy
* Pandas 
* Matplotlib
* Sklearn



========================================================================================================================

(1) Time series conversion function
-----------------------------------

Summary 
-------
Reframes time series data as a sliding window for supervised learning -
Returns a modified DataFrame object 


Arguments
---------
Function takes the following arguments:

	- data -> Pandas DataFrame object, univariate or multivariate time series data 

	- n_in -> int, number of lag steps in the past to shift 

	- n_out -> int, number of forecast steps in the future to shift 

	- dropnan -> bool, if True removes rows containing null values (default = True)


Example
--------

obs1 = [x for x in range(10)]
obs2 = [x for x in range(50,60)]

df = pd.DataFrame({'obs1':obs1,'obs2':obs2})


data = timeser_convert(df,1,2)


print(data)
#   var1(t-1)  var2(t-1)  var1(t)  var2(t)  var1(t+1)  var2(t+1)    
#1        0.0       50.0        1       51        2.0       52.0          
#2        1.0       51.0        2       52        3.0       53.0           
#3        2.0       52.0        3       53        4.0       54.0           
#4        3.0       53.0        4       54        5.0       55.0          
#5        4.0       54.0        5       55        6.0       56.0           
#6        5.0       55.0        6       56        7.0       57.0           
#7        6.0       56.0        7       57        8.0       58.0          




========================================================================================================================

(2) LSTM model
--------------

-------
Summary
-------
One LSTM layer, 50 hidden nodes

One Dense layer - fully connected layer, corrects dimensionality of desired target for output.  Corresponds to equation y^t = W * h^t

Dropout - regularization method where input and recurrent connections are probabilistically excluded from activation and weight 
updates during training. Prevents overfitting (* removed for performance in single layer model)

Optimizer and metrics - Adam optimizer, mean_square_error and mean_absolute_error metrics to evaluate performance during training, 
Nash-Sutcliffe's Efficiency to evaluate test set prediction accuracy after training.

EarlyStopping - allows use of arbitrary (large) number of epochs in training, and model will break out of training when model has 
failed to produce a lower loss value in n number of epochs (specified by keyword argument patience). Prevents under and overtraining.


----------
Arguments
----------
The model is nested in a function that takes the following arguments: 

	- data -> Pandas Dataframe

	- read_cols -> list of column names  * NOTE the column of values you want to predict MUST be placed at the front of the list

	- lag_step -> number of time steps in the past (these will be the inputs to your model)

	- forecast_step -> number of time steps in the future to forecast (this will be your predicted variable)

	- train_split -> number of hours to train on

	- hidden_nodes -> number of hidden nodes in LSTM layer

	- n_epochs -> number of epochs for training (model may stop before this number is reached

	- batch_size -> batch size

	- graph_file -> file path to send graphs of results  

	- csv_file -> file path to send csv of results


----------------
Data Preparation 
----------------

(1) Pandas DataFrame is converted to Numpy array containing all values for all variables, including your target (column of values you want to predict)
(2) All data is normalized to range 0-1
(3) Values are reframed for sliding window training and predictions using timeser_convert function (explained above)
(4) Values are split into train and test sets at the user-specified number of hours
(5) Train and test sets are then each split again, this time into X's (model inputs) and y's (the values you want it to predict from the inputs)
(6) Train X and test X are reshaped into 3D tensors (with dimensions - [samples, timesteps, features].  << NOTE at this step, if you've removed your target as an input to the models, features dimension in the reshape method call should be changed to features-1

These are the values you will give the model to use to train and predict your target values (y's)

========================================================================================================================

(3) Sensitivity Analysis 
------------------------

-------
Summary
-------
Nested for loop. 
Allows performance testing on a range of parameters and inputs.
  
Purpose is to determine optimal parameters and importance of each input to determining magnitude of discharge/flood event.


--------------
User-Specified
--------------
At each level of the nested loop, user's can specify a range of: 

	- inputs and combinations of inputs (from read_cols)
	
	- lag steps 

	- forecast steps

	- train hours 

	- times the model should run using each set of parameters and inputs


Other settings can be specified in parameters dictionary (called with model(**parameters))


-------
Example
-------

	     TARGET
read_cols=['variable1','variable2','variable3'...]

baseDir = 'insert/path/to/a/new/base/directory/to/store/all/results'



for INDEX in range(1,len(read_cols)+1):
    

    ### ITERATE THROUGH EACH COMBO OF INPUTS - 1, 1&2, 1&2&3...
    ------------------------------------------------------------
    COLS = read_cols[:INDEX]. 

    for FORECAST_STEP in range(2,14): <<< 1 - 12 steps in the future 
	
	### CREATE SUBDIRECTORIES TO HOLD RESULTS FOR EACH LOOP/COMBINATION OF SETTINGS
	-------------------------------------------------------------------------------
        dirname = str(COLS)+'_'+ 'forecast ' + str(FORECAST_STEP)
        subdir = os.path.join(baseDir,dirname)
        os.mkdir(subdir)

        for RUN in range(1,4):   <<< three runs for each set of parameters
	    
	    ### CREATE NEW FILES IN SUBDIRECTORY
 	    ------------------------------------
            graph_file = open(subdir+ f'/GRAPH_{dirname}_run{run}.png','wb')
            csv_file = open(subdir + f'/CSV_{dirname}_run{run}.csv','w')
	

	    ### RUN MODEL WITH EACH COMBINATION OF SETTINGS
	    -----------------------------------------------
	    model_loop(**parameters,read_cols=cols,forecast_step=step,graph_file=graph_file,csv_file=csv_file)                            
		
            plt.close()




========================================================================================================================

(4) USGS data extraction, resample, merge, and export
-----------------------------------------------------

-------
Summary
-------
Three functions
Can be used to compile data for other watersheds to enhance model robustness


1) df_usgs('url','time_interval')
----------------------------------
Creates a datetime-indexed Series from tab-delimited USGS Surface Water Data

	Accepts two arguments: 

	       'url' ->  Points to page in USGS database containing tab-delimited data for ONE PARAMETER measured values in user-specified data range
	  		    
	       'time_interval' -> Function resample time series to this interval, taking the mean of values between. 
				  Ex. 'H' or '30min' or 'D'	    


2) export(*args,xlsx)
---------------
Exports Series/Dataframes to separate tabs in an excel document
(Document must already exist in your working directory)


	Accepts two arguments: 

		*args -> lists, each containing one Series/Dataframe and a corresponding sheet name

		xlsx -> excel file destination 


3) merge(merge_file,'regex_nameMatch')
----------------
Reads Excel doc sheet names and contents merges contents of sheets with shared watershed name
Returns a Dataframe of all columns for your selected site
	






---------
Example
---------
For this example, Radecke and Todd's Avenue gauges are located in the same watershed (Moore's Run)
Rocky Branch is located in a different watershed 




1) Extract data for each site and resample to desired time interval
--------------------------------------------------------------------
- df_usgs('url','time_interval')



Radecke_ave = df_usgs('https://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&format=rdb&site_no=01585230&period=&begin_date=2003-06-01&end_date=2003-07-01')


                       Discharge (cfs)     < correct field name will replace USGS parameter code   
#Timestamp                          
#2003-06-01 00:00:00            2.00
#2003-06-01 00:01:00            2.00	   < NOTE 
#2003-06-01 00:02:00            2.00	   < minute 
#2003-06-01 00:03:00            2.00       < intervals 
#2003-06-01 00:04:00            2.00



Todds_ave = df_usgs('https://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&format=rdb&site_no=01585225&period=&begin_date=2003-05-01&end_date=2003-06-01')

                     Discharge (cfs)            
Timestamp                          
2003-05-01 00:00:00            0.05
2003-05-01 00:01:00            0.05	  
2003-05-01 00:02:00            0.05
2003-05-01 00:03:00            0.05
2003-05-01 00:04:00            0.05


Rocky_branch_precip = df_usgs('https://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00045=on&format=rdb&site_no=0208735012&period=&begin_date=2010-03-01&end_date=2010-05-31')

                    Precipitation	
Timestamp                                   
2010-03-01 00:00:00          0.00
2010-03-01 00:15:00          0.00
2010-03-01 00:30:00          0.00
2010-03-01 00:45:00          0.00
2010-03-01 01:00:00          0.00





2) Export each dataset to Excel sheet
  (NOTE you must include shared watershed names in sheet_names of Series you want to merge)
------------------------------------------------------------------------------------------
- export([Series,sheet_name],xlsx)       
				       
				       
*FUNCTION CALL*			       
---------------		          ------------				------------
export([Radecke_ave, 'Radecke_Ave_(Moores_Run)', Todds_ave, 'Todd's_Ave_(Moores_Run)',[Rocky_branch_precip, 'Rocky_branch_precip'],your_Excelfile.xlsx)
				   ----------				 ----------
					                              	      

*PRINTS*
---------
Your working directory: /path/to/your/working_directory/
Exporting to /your_Excelfile.xlsx

  0%|          | 0/3 [00:00<?, ?it/s]
 67%|██████▋   | 2/3 [00:00<00:00, 13.54it/s]
100%|██████████| 3/3 [00:00<00:00,  8.52it/s]

Data exported






3) Merge data from gauges in the same watershed 
---------------------------------------------------
merge(xlsx,watershed_ID,fillnan=True)
	   ------------
		  ^ 
		  ^ 
		  NOTE - Use a unique ID, regex will match and merge with other Series in xlsx carrying that ID


*FUNCTION CALL*
---------------
merged = merge(collect_file,'Moores')


*PRINTS*
-----------
Radecke_Ave_(Moores_Run)
 match
Todd's_Ave_(Moores_Run)
 match
Rocky_branch_precip



*RETURNS*
-----------

              Timestamp  Radecke_Ave_(Moores_Run)_Discharge (cfs)  \
 0 2003-05-31 23:30:00                                  2.000000   
 1 2003-06-01 00:00:00                                  1.886667   
 2 2003-06-01 00:30:00                                  1.726667   
 3 2003-06-01 01:00:00                                  1.570000   
 4 2003-06-01 01:30:00                                  1.456667   
 
    Todd's_Ave_(Moores_Run)_Discharge (cfs)  
 0                                      0.1  
 1                                      0.1  
 2                                      0.1  
 3                                      0.1  
 4                                      0.1  






