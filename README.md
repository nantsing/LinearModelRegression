# LinearModelRegression
This project contains four models: linear regression, ridge regression, RBF kernel rgression and lasso regression.
1. All four models code is in model.py
2. Training and testing data is in train.py
3. test.py is for calculating correlation coefficient and determination coefficient
4. data_preprocess.py should have been used to normalize data. And since this part of code is implemented in dataset.py, data_preprocess.py has never been used.

To see reproduce the results in the report, run the command in a linux teminel:

    python train.py

In train.py, training process is default off and it will always load the pretrained model in models. You can change it by setting

    model_train = True

And you can change other setting to get other results. 

To calculate correlation coefficient and determination coefficient of training data, run the command in a linux teminel:

    python test.py

If you have any problem with the code, please feel free to contact me. 