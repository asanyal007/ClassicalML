from imblearn.over_sampling import SMOTE

def apply_smote(X,y):
    print("Before OverSampling, counts of label '1': {}".format(sum(y==1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y==0)))
    sm = SMOTE(random_state=2)
    X_s, y_s = sm.fit_resample(X, y)
    print("After OverSampling, counts of label '1', %: {}".format(sum(y_s==1)))
    print("After OverSampling, counts of label '0', %: {}".format(sum(y_s==0)))

    return X_s, y_s

def appy_undersample(X, y) :
    from imblearn.under_sampling import RandomUnderSampler
    ros = RandomUnderSampler(random_state=42, sampling_strategy = {0: 18213, 1: 8213})
    print("Before OverSampling, counts of label '1': {}".format(sum(y==1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y==0)))
    X_us, y_us = ros.fit_resample(X, y)
    print("after OverSampling, counts of label '1': {}".format(sum(y_us==1)))
    print("after OverSampling, counts of label '0': {} \n".format(sum(y_us==0)))

    return X_us, y_us 