import numpy as np
import pandas as pd

import tensorflow as tf

from utils import Utils

class Models:
         
    def model_creation(self,):
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(8,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(150,activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(150,activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1,activation='sigmoid')
            
        ])
        
        model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )
        
        print(model.summary())
        
        return model
        
        
    def model_training(self,X_train,X_test,y_train,y_test):
        model = self.model_creation()
        
        model.fit(X_train,y_train,
                  validation_data=(X_test,y_test),
                  batch_size=512,
                  epochs=100
                  )
        utils = Utils()
        utils.model_exports(model)