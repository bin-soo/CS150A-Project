import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,GroupKFold
import os
from dataSeperation import sample1,trainx1,trainy1,trainxs,trainys
path = os.getcwd()
from keras import backend
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

print(trainx1.head())
print(trainx1.shape)
print(trainy1.head())
print(trainy1.shape)
trainx = trainx1
trainy = trainy1
rmseList = []

for kfold, (train, val) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42)\
                                              .split(trainx, trainy)):

    X_train, X_val = trainx.iloc[train], trainx.iloc[val]
    y_train, y_val = trainy.iloc[train], trainy.iloc[val]

    fcnn = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2,activation='softmax')
    ])

    fcnn.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.3),
                 loss = 'sparse_categorical_crossentropy',
                 metrics=[rmse])

    history = fcnn.fit(X_train,y_train,
                       epochs=10,
                       batch_size = 32,
                       shuffle = True,
                       validation_data = (X_val,y_val)).history
    rmseList.append(history['rmse'])
    fcnn.save(path)

'0.53~0.61'
'''origin 256, 64, 32, 2'''