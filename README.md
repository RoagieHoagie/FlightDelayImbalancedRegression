# FlightDelay
Data analysis of US Flight Delays based on the Flight Delay kaggle dataset by Arvind Nagaonkar
## Before using the script, make sure to have the database installed: 
https://www.kaggle.com/datasets/arvindnagaonkar/flight-delay
(access 4.05.2026)
## Initial preprocessing (done)
note: date in cyclically encoded formatting
## Imbalanced preprocessing
## Modelling and evaluation
note: if needed, target encoding for 
'originCity', 'originState', 'destCity', 'destState', 'carrierName'
should be performed after split to prevent data leakage. For now they've been converted to categorical