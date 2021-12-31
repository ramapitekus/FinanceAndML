* Install requirements.txt

* Run main.py from data_pipeline folder
    * scrapes the data from bitinfocharts.com
    * Afterwards, data is cleaned (NAs are linearly interpolated wherever possible, fills with most common values otherwise)
    * feature selection based on random forests is performed
    * VIF is used to reduce multicollinearity
    * Time intervals, indicators etc. are specified in settings.json.

* Regression folder contains ML models to predict prices in specified periods (1, 7, 30, 90 days future price)
    * Models used: ANN, Stacked ANN, SVM, LSTM
    * Models folder contains trained models for NNs
    * logs contain output of training, as well as statistics to the trained models