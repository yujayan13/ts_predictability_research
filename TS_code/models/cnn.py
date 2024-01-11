import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
import scipy.stats as measures

import keras.backend as K

#define relative squared error loss function
def relative_squared_error(y_true,y_pred):
  mse = tf.keras.losses.MeanSquaredError()
  mean_squared_error = mse(y_true,y_pred)
  var = mse(y_true,K.mean(y_true))
  rse = mean_squared_error/var
  return(rse)

#pairwise shuffling of inputs and outputs
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

months = ["march","april","may","june","july","august"]

#load data
data_dict = {}
for month in months:
    
    month_data = np.load(f"C:\ts_research\training_{month}_ts.npz")
    data_dict[month] = month_data
    
#set random seed for reproducibility
seed = 28
tf.random.set_seed(seed)

#training models for each month
for month in months:
    month_MSE = []
    month_RMSE = []
    month_x_tests = []
    month_y_tests = []
    month_correlations = []
    month_predictions = []
    
    #loading inputs (inputs will be the same regardless of month)
    inputs = data_dict[month]['enso'].reshape((6400,5,22,57,1))

    output = np.array([data_dict[month]["ts"]], dtype=np.float64).reshape(6400,691)
    #shuffle data
    inputs, output = unison_shuffled_copies(inputs,output)
    DATASET_SIZE=len(inputs)
    inputs = tf.convert_to_tensor(inputs)
    
    for point in range(691):
        #converts numpy array to tensors, selecting outputs (TS for each gridpoint)
        outputs = tf.convert_to_tensor(np.array([i[point] for i in output], dtype=np.float64))

        #splits into training and testing data in a 70:15:15 ratio
        TRAIN_SIZE=int(7*DATASET_SIZE/10)
        VAL_SIZE = int(15*DATASET_SIZE/100)
        TEST_SIZE = int(15*DATASET_SIZE/100)
        x_train=inputs[:TRAIN_SIZE]
        y_train=outputs[:TRAIN_SIZE]
        x_val=inputs[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
        y_val=outputs[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
        x_test=inputs[TRAIN_SIZE+VAL_SIZE:]
        y_test=outputs[TRAIN_SIZE+VAL_SIZE:]
        
        #build model
        #The hyperparameters were chosen through a gridsearch
        model = models.Sequential()
        model.add(layers.Conv3D(32, (5,3,3), activation="relu", input_shape=(5,22,57,1))) #Conv3d is used to process the timeseries where the time is the third axis
        model.add(tf.keras.layers.Reshape((20,55,32))) #reshape because max pooling cannot be done for 3d
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, 3, activation="tanh")) #once the timeseries is processed, we can use Conv2d again
        model.add(layers.Flatten())
        model.add(layers.GaussianNoise(0.1, seed=28))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(1))
        #use custom loss function relative squared error
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                    loss=relative_squared_error,
                    metrics=[tf.keras.metrics.RootMeanSquaredError()])
        
        #early stopping when model stops learning
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=250,
            verbose=0,
            mode="min",
            baseline=None,
            restore_best_weights=True,
        )
        
        #run model
        history = model.fit(x_train,y_train, epochs=750, callbacks = [callback],
                            validation_data=(x_test,y_test), batch_size = 64)
        print(history)
        
        m,r = model.evaluate(x_test, y_test, batch_size=64)


        month_RMSE.append(r)
        p = []

        p.append(model.predict(x_test))
        
        if p[1:] == p[:-1]:
          p[0]+=0.0001
        per_coef = measures.pearsonr(np.array(p).flatten(), y_test.numpy())[0]
        month_correlations.append(per_coef)
        month_y_tests.append(y_test)
        month_predictions.append(p)
        print(point)
        model.save('model'+str(point)+month+'.hdf5')
    
    month_x_tests.append(x_test)
    

    np.savez(f"results_{month}_ts", RMSE=np.array(month_RMSE),corr_coeffs = np.array(month_correlations), x_test = np.array(month_x_tests), y_test=np.array(month_y_tests),pred = np.array(month_predictions))
