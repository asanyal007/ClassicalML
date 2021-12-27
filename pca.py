from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import pandas as pd

class getPCA:
    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)
        self.data = None 

    def apply_StandardScaler(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        scaled = scaler.transform(data)
        
        return scaled

    def apply_MaxAbsScaler(self, data):
        scaler = MaxAbsScaler()
        scaler.fit(data)
        scaled = scaler.transform(data)

        return scaled

    def apply_RobustScaler(self, data):
        scaler = RobustScaler()
        scaler.fit(data)
        scaled = scaler.transform(data)

        return scaled
    

    
    def apply_pca(self, scaled_data):
        self.pca.fit(scaled_data)
        x_scaled = self.pca.transform(scaled_data)
        d1 = pd.DataFrame(x_scaled)
        return d1

        #final_data = pd.concat([d1,data1.dropna()['isFraud']], axis=1)


