#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 20:35:04 2022

@author: mohamaddalati
"""

### Assignment 2: Mohamad Dalati 

# Load libraries
from sklearn.neighbors import KNeighborsClassifier #used for KNN
import numpy as np
import pandas as pd

## Task 2 

# Create a dataframe   
data = np.array( [ ['Black', 1, 1] , ['Blue', 0, 0], ['Blue', -1, -1] ] ) 
column_names = ['y', 'x1', 'x2']
row_names = [ 'A', 'B', 'C' ]
df = pd.DataFrame(data, columns = column_names, index = row_names) 
X = df.iloc[: , 1:3]
y = df['y']
# Keep in mind that the values are not standardized YET

#Create the new observation and combine it to the Training Set 
new_obs = pd.DataFrame( [ ['D', 0.1, 0.1] ], columns = ['Index', 'x1', 'x2' ]  )
new_obs.set_index("Index", inplace = True) # setting the index variable as the true index
combined_obs = X.append(new_obs) # Combining the new variables atto the Training set 
#                                  (i.e. the predictors df named X)


# Standardize the combined dataset using Z-score Standardization 
# The idea here is to standardize the training set and new observations together and then split them again!
from sklearn.preprocessing import StandardScaler # StandardScaler is always used for z-score standardization
standardizer = StandardScaler()
combined_obs_std = standardizer.fit_transform(combined_obs)


# Split the data after standardization (split the data to the training set and new observation "AFTER" they have 
# been standadized! )
X_std = combined_obs_std[:3, :] # To retrieve the "Standardized Training set" back  
new_obs_std = combined_obs_std[3:,:] # To retrieve the "Standardized new observed set" back

# Build a model with k = 2 
knn2 = KNeighborsClassifier(n_neighbors = 2) # Without any other parameters  
model2 = knn2.fit(X_std, y)

###         Task 4. Make a prediction for a new observation 
model2.predict(new_obs_std) #Very Important note below! (summary)
# The answer was classified as " Black "


###         Task 5. 
model2.predict_proba(new_obs_std) #this give the probability of each target variable 




    
    
    
    
    
    
    
    
