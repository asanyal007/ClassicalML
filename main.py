import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import pca
import smote
import SVM
#import lime.lime_tabular
data = pd.read_csv('data/data.csv')

pca_obj = pca.getPCA(n_components=4)

# convert categorical type column to binary class
data1 = pd.get_dummies(data, columns=['type'])

my_file = Path("pca_data/pca_data.csv")
if my_file.is_file():
    print("PCA trnasformed data already exists!")
    final_data = pd.read_csv('pca_data/pca_data.csv').drop(['Unnamed: 0'], axis=1)
    print(final_data.head())
else:
    # remove class columns and un-necessery cols
    data2 = data1.drop(['nameOrig', 'nameDest','isFraud'], axis=1)
    scaled = pca_obj.apply_scale(data2)
    x_scaled_pca = pca_obj.apply_pca(scaled)
    
    final_data = pd.concat([x_scaled_pca,data1['isFraud']], axis=1)
    final_data.to_csv('pca_data/pca_data.csv')

X=final_data.drop(['isFraud'], axis=1)
y=final_data['isFraud']

X_us , y_us = smote.appy_undersample(X,y, {0: 20000, 1: 8213})
X_os, y_os = smote.apply_smote(X_us,y_us)

X_train, X_test, y_train, y_test = train_test_split(X_os, y_os, test_size = 0.3, random_state = 42)
#apply SMOTE
#X_train_s, y_train_s, X_test_s, y_test_s = smote.apply_smote(X_train, y_train, X_test, y_test)

# train and get predition
obj_SVM = SVM.getSVM(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, C=1,  kernel='rbf', random_state=0)
obj_SVM.apply_svm()
obj_SVM.save_model()

con_mat, average_precision, cls_report = obj_SVM.get_performance_metrix()
print("con mat", con_mat)
print("precission", average_precision)
print("cls report", cls_report)
#pca_obj.apply_scale(data)


