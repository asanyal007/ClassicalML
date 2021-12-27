from imblearn.over_sampling import SMOTE

def apply_smote(X_train, y_train, X_test, y_test):
    print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
    sm = SMOTE(random_state=2)
    X_train_s, y_train_s = sm.fit_resample(X_train, y_train)
    X_test_s, y_test_s = sm.fit_resample(X_test, y_test)
    print("After OverSampling, counts of label '1', %: {}".format(sum(y_train_s==1)/len(y_train_s)*100.0,2))
    print("After OverSampling, counts of label '0', %: {}".format(sum(y_train_s==0)/len(y_train_s)*100.0,2))

    return X_train_s, y_train_s, X_test_s, y_test_s