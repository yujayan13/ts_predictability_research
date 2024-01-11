import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import scipy.stats as measures

import random
import keras.backend as K


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#only using march data for the gridsearch
data = np.load(r"C:\ts_research\training_march_ts.npz")

seed = 28
tf.random.set_seed(seed)

inputs = data['enso'].reshape((6400,5,22,57,1))
#converts numpy array to tensors

output = np.array([data["ts"]], dtype=np.float64).reshape(6400,691)
#shuffle data
inputs, output = unison_shuffled_copies(inputs,output)
DATASET_SIZE=len(inputs)
inputs = tf.convert_to_tensor(inputs)

def relative_squared_error(y_true,y_pred):
  mse = tf.keras.losses.MeanSquaredError()
  mean_squared_error = mse(y_true,y_pred)
  var = mse(y_true,K.mean(y_true))
  rse = mean_squared_error/var
  return(rse)

#function for building and training model with arbitrary hyperparameters
def create_model(epoch, activation1, activation2, activation3, activation4, filter1, filter2, gaussian, lr, patience, restore, batch):
    model = models.Sequential()
    #Gaussian noise layer to prevent overfitting
    #model.add(layers.GaussianNoise(0.1, seed=None,input_shape=(5,10, 41,1)))
    model.add(layers.Conv3D(filter1, (5,3,3), activation=activation1, input_shape=(5,22,57,1))) #Conv3d is used to process the timeseries where the time is the third axis
    model.add(tf.keras.layers.Reshape((20,55,filter1))) #reshape because max pooling cannot be done for 3d
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filter2, 3, activation=activation2)) #once the timeseries is processed, we can use Conv2d again
    model.add(layers.Flatten())
    model.add(layers.GaussianNoise(gaussian, seed=None))
    model.add(layers.Dense(64, activation=activation3))
    model.add(layers.Dense(32, activation=activation4))
    model.add(layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
                loss=relative_squared_error,
                metrics=[tf.keras.metrics.RootMeanSquaredError()])


    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        verbose=0,
        mode="min",
        baseline=None,
        restore_best_weights=restore,
    )

    history = model.fit(x_train,y_train, epochs=epoch, callbacks = callback,
                        validation_data=(x_test,y_test), batch_size = batch)
    print(history)
    m,r = model.evaluate(x_test, y_test, batch_size=batch)

    p = []

    p.append(model.predict(x_test))
    
    if p[1:] == p[:-1]:
      p[0]+=0.0001
    
    per_coef = measures.pearsonr(np.array(p).flatten(), y_test.numpy())[0]
    return(per_coef, r, p)

batch_sizes = [32,64,256,512]
epochs = [250,500,750,1000]
learning_rates = [0.001,0.01,0.1]

filters = [[32,16],[64,32],[128,64],[256,128]]


activation1 = ["tanh", "relu"]
activation2 = ["tanh", "relu"]
activation3 = ["tanh", "relu"]
activation4 = ["tanh", "relu"]


pats = [100,250,500]
restore = [True, False]

gaussians = [0.001, 0.01,0.1]


best_f1_list = []
best_f2_list = []
best_a1_list = []
best_a2_list = []
best_a3_list = []
best_a4_list = []
best_pats_list = []
best_restore_list = []
best_gaussian_list = []
best_epochs_list = []
best_lr_list = []
best_batch_list = []


best_pearsons = []
best_rmses = []
saved_rmses = []

points = []

#repeat 3 times for 3 randomly sampled points and choose optimal hyperparameters
for t in range(3):
    if t == 0:
        point = 0
    else:
        point = random.randint(0,690)
    points.append(point)
    outputs = tf.convert_to_tensor(np.array([i[point] for i in output], dtype=np.float64))

    #splits into training and testing data in a 50:50 ratio
    TRAIN_SIZE=int(7*DATASET_SIZE/10)
    VAL_SIZE = int(15*DATASET_SIZE/100)
    TEST_SIZE = int(15*DATASET_SIZE/100)
    x_train=inputs[:TRAIN_SIZE]
    y_train=outputs[:TRAIN_SIZE]
    x_val=inputs[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
    y_val=outputs[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
    x_test=inputs[TRAIN_SIZE+VAL_SIZE:]
    y_test=outputs[TRAIN_SIZE+VAL_SIZE:]

    
    best_a1 = 0
    best_a2 = 0
    best_a3 = 0
    best_a4 = 0
    best_f1 = 0
    best_f2 = 0
    best_lr = 0
    best_RMSE = 999999
    saved_RMSE = 0
    best_pearson = 0

    #iterate through hyperparameters. in the code below, we only iterate through activation functions, filters, and learning rates as other hyperparameters were selected through another gridsearch
    for a1 in activation1:
      print(a1)
      for a2 in activation2:
        for a3 in activation3:
          for a4 in activation4:
            for f in filters:
                for lr in learning_rates:
                    pearson,RMSE,p = create_model(750, a1, a2, a3, a4,f[0],f[1],0.1,lr,250,True,64)
                    #compare skill score after model training
                    if pearson>best_pearson:
                        best_a1 = a1
                        best_a2 = a2
                        best_a3 = a3
                        best_a4 = a4
                        best_f1 = f[0]
                        best_f2 = f[1]
                        best_lr = lr
      
                        best_pearson = pearson
                        saved_RMSE = RMSE
                    if RMSE<best_RMSE:
                        best_RMSE = RMSE
    best_a1_list.append(best_a1)
    best_a2_list.append(best_a2)
    best_a3_list.append(best_a3)
    best_a4_list.append(best_a4)
    
    best_f1_list.append(best_f1)
    best_f2_list.append(best_f2)
    
    best_lr_list.append(best_lr)
    
    best_pearsons.append(best_pearson)
    best_rmses.append(best_RMSE)
    


print("a1: " + str(best_a1_list))
print("a2: " + str(best_a2_list))
print("a3: " + str(best_a3_list))
print("a4: " + str(best_a4_list))
print("f1: " + str(best_f1_list))
print("f2: " + str(best_f2_list))
print("lr: " + str(best_lr_list))



print(saved_rmses)
print(best_rmses)
print(best_pearsons)