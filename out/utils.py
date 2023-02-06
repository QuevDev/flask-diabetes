import pandas as pd
import numpy as np
import joblib 

class Utils:
    
    def load_data(self, path):
        df = pd.read_csv(path)
        df = pd.DataFrame(np.repeat(df.values,repeats=10,axis=0),columns=df.columns)
        return  df
    
    def featues_target(self,data,target):
        X = data.drop([target],axis=1)
        y = data[target]
        
        return X,y 
    
    def train_test(self,test_size,x,y):
        x_len = len(x)
        test_size = int(x_len * test_size)
        
        X_train = x[test_size:]
        X_test = x[:test_size]
        
        y_train = y[test_size:]
        y_test = y[:test_size]
        
        return X_train,X_test,y_train,y_test
    
    
    def model_exports(self,clf):
        joblib.dump(clf,'../models/diabetes_model.pkl')
            
        
        