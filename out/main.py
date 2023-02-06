from utils import Utils
from models import Models

if __name__ == '__main__':
    
    utils = Utils()
    models = Models()
    
    df = utils.load_data('../in/diabetes.csv')
    
    X,y = utils.featues_target(df,'Outcome')
    
    #datos de train y test
    X_train,X_test,y_train,y_test = utils.train_test(0.2,X,y)
    models.model_training(X_train,X_test,y_train,y_test)
    
    