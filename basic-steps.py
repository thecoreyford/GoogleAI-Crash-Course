#basic-steps.py

#================================================================
#INPUT_FUNCTION
# convert pandas data into dict of np arrays

# make dataset and get batches (for num_epocs)

# Shuffle data?

# return the next batch (as tuple)

#================================================================
#TRAIN_MODEL

# get the dataset 
#https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv

# shuffle the dataset

# scale to be in units of 1000s so learning rates are more "typical"


# Remember tensorflow has its own data structure for... 
# ..column's (both numeric and categorical)
# with this knowledge get features and targets 


# create input functions (predict and training)


# Use gradient descent optmizer within a linear regressor (clip gradient descent)...
# ... or make regressor!


# train linear regressor (in loop to see progress)
#		do training 
# 		compute predictions
#		show root mean squared 


#================================================================
#CALL_THE_MODEL

