import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
#from mlxtend.plotting import plot_confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class getSVM:
    def __init__(self, X_train, X_test, y_train, y_test , C,  kernel, random_state) :
        self.X_train =X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.classifier= svm.SVC(C= C, kernel= kernel, random_state= random_state)
    
    def apply_svm(self ):
        self.classifier.fit(self.X_train, self.y_train)
        self.y_pred = self.classifier.predict(self.X_test)

        #return self.y_pred

    def save_model(self):
        filename = 'model/SVM.sav'
        pickle.dump(self.classifier, open(filename, 'wb'))
    def load_model(self):
        filename = 'model/SVM.sav'
        self.classifier = pickle.load(open(filename, 'rb'))

    def get_performance_metrix(self):
        con_mat = confusion_matrix(self.y_test, self.y_pred)
        average_precision = average_precision_score(self.y_test, self.y_pred)
        cls_report = classification_report(self.y_test, self.y_pred)

        return con_mat, average_precision, cls_report
    

        