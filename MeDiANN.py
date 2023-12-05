import numpy as np 
import tensorflow as tf
from tensorflow import keras 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from constants import * 
import pandas as pd
from util import * 

class MeDiANN_Magic():
    def __init__(self,data,target='R',model_path=None,experiment_name='test', root_path  = None, 
                 pre_trained_model_path = None,output_path=None,activation_function = ACTIVATION_FUNCTION, 
                 output_type = OUTPUT_FILE_TYPE, lite_model_option = False, epochs = NUM_EPOCHS):
        """
        This class will be used to deal with the training, prediction and storing the results of 
        the MeDiANN algorithm

        -----------------------------------------------------------------------------------------
        Input:

        data : The data dictionary, generated from the data importer
        model_path: The model where you want the models to be stored
        experiment_test: The name that you will use for this test, all the files will be named using this name
        root_path : The root path where you want the result and model of your experiment to be stored
        pre_trained_model_path: This is the path for your pretrained model, if you have one
        output_path: This is where you are going to store the result of your model
        activation_function: This is the activation function. The optimal one that we found is ReLU
        output_type: The type of the output. It can be .csv or .npy
        lite_model_option: Apply the lite version to prevent overfitting and made the training faster
        
        -----------------------------------------------------------------------------------------


        """
        self.input = data['X']
        self.target = 'R'
        self.output = data[target]
        self.train_set = data['train_set']
        self.val_set = data['val_set']
        self.test_set  = data['test_set']
        self.scalers = data['scalers']
        self.experiment_name = experiment_name
        if model_path == None:
            model_path = root_path + '/model/'
            check_if_data_folder_exists(model_path)
        self.model_path = model_path
        if output_path == None:
            output_path = root_path + '/output'
            check_if_data_folder_exists(output_path)
        self.output_path= output_path
        self.activation_function = activation_function
        self.output_type = output_type
        self.pre_trained_model_path = pre_trained_model_path
        self.lite_model = lite_model_option
        self.epochs = epochs


    def mediann_model(self):
        """

        Definition of the full MeDiANN Neural Network architecture.
        To be used for small output (e.g. 101/202 values as an output)
        
        """
        dim_output = self.output.shape[1]
        dim_input = self.input.shape[1]
            # Define the model architecture
        model = keras.Sequential([
            keras.layers.Dense(dim_output*4, activation=self.activation_function, input_shape=(dim_input,)),
            keras.layers.Dense(dim_output*8, activation=self.activation_function),
            keras.layers.Dense(dim_output*16, activation=self.activation_function),
            keras.layers.Dense(dim_output*8, activation=self.activation_function),
            keras.layers.Dense(dim_output*4, activation=self.activation_function),
            keras.layers.Dense(dim_output*2, activation=self.activation_function),
            keras.layers.Dense(dim_output)
        ])
        # Print the model summary
        print('The model has been defined \n')
        #Set loss function and optimizer
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model


    def mediann_model_lite(self):
        """

        Definition of the full MeDiANN Neural Network architecture.
        To be used for large output (e.g. 501/1002 values as an output)
        
        """
        dim_output = self.output.shape[1]
        dim_input = self.input.shape[1]
            # Define the model architecture
        model = keras.Sequential([
            keras.layers.Dense(dim_output*2, activation=self.activation_function, input_shape=(dim_input,)),
            keras.layers.Dense(dim_output*4, activation=self.activation_function),
            keras.layers.Dense(dim_output*8, activation=self.activation_function),
            keras.layers.Dense(dim_output*4, activation=self.activation_function),
            keras.layers.Dense(dim_output*2, activation=self.activation_function),
            keras.layers.Dense(dim_output)
        ])
        # Print the model summary
        print('The model has been defined \n')
        #Set loss function and optimizer
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model


    def predict_using_pretrained_model(self):
        """

        To be used if we want to apply our predictions without doing any training
        cause we have a pretrained model
        
        """
        try:
            X = np.array(self.input).astype(float)
            model = tf.keras.models.load_model(self.pre_trained_model_path)
            pred = model.predict(X)
            if self.output_type!='npy':
                pred = pd.DataFrame(pred)
                pred.to_csv(self.output_path + '/pred_from_pretrained_experiment'+self.experiment_name+'.csv')
            else:
                np.save(self.output_path+'/pred_from_pretrained_experiment'+self.experiment_name+'.npy',pred)
        except:
            print('The pretrained model is not working, ignoring this step...\n')
            pred = None
            model = None
        return model,pred
    

    def predict(self,model,model_name='baseline_method'):
        """
        Function to use to predict the values once that the model has been trained. 
        This is a function that we use if we don't want to apply any boost.
        -----------------------------------------------------------------------------------------
        Input:

        model : The keras object that we want to use to predict our output
        model_name : The name of the model. We will use this to save our data
        
        -----------------------------------------------------------------------------------------

        Output:

        pred : The prediction + the prediction saved as a file

        """
        model_name = '/'+model_name+'_'+self.experiment_name+'_pred'
        print('The model results will be saved as a %s file with name %s'%(self.output_type,model_name))
        pred = model['model'].predict(self.input)
        if self.output_type == 'npy':
            np.save(self.output_path+model_name+'.npy',pred)
        else: 
            pred = pd.DataFrame(pred)
            pred.to_csv(self.output_path+model_name+'.csv')
        return pred

    def predict_boost(self,model_pred,model_boost,model_name='boost_method'):

        """
        Function to use to predict the values once that the model and its boost have been trained. 
        -----------------------------------------------------------------------------------------
        Input:

        model_pred : The prediction of the baseline model
        model_boost : The keras object that has been trained on the difference, as a boost model
        model_name : The name of the model. We will use this to save our data
        
        -----------------------------------------------------------------------------------------

        Output:

        pred : The prediction + the prediction saved as a file

        """


        model_name = '/'+model_name+'_'+self.experiment_name+'_pred'
        print('The boost model results will be saved as a %s file with name %s'%(self.output_type,model_name))
        pred = model_pred+model_boost['model'].predict(self.input)
        if self.output_type == 'npy':
            np.save(self.output_path+model_name+'.npy',pred)
        else: 
            pred = pd.DataFrame(pred)
            pred.to_csv(self.output_path+model_name+'.csv')
        np.save(self.output_path+model_name,pred)
        return pred
        

    def train_baseline_model(self):
        """
        To be used to train a baseline model
        """

        X = np.array(self.input).astype(float)
        Y = np.array(self.output).astype(float)
        if self.lite_model == True:
            model = self.mediann_model_lite()
        else:
            model = self.mediann_model()
        #Building the training input
        X_train, X_val = X[self.train_set],X[self.val_set]
        #Building the validation input 
        Y_train, Y_val = Y[self.train_set],Y[self.val_set]
        model_name = 'baseline_model_'+self.target+'_'+self.experiment_name+'.hdf5'
        print('Training of the baseline machine learning model has started \n')
        #Check the validation loss and stop when the model is overfitting
        #Save the best model as a h5 file
        save_model_name = self.model_path+'/'+model_name
        model_checkpoint = ModelCheckpoint(save_model_name, monitor='val_loss', save_best_only=True)
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=20, factor=0.5, min_lr=1e-5)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        #Store the history of your model
        history = model.fit(X_train,Y_train,validation_data = (X_val,Y_val), epochs = self.epochs, batch_size = BATCH_SIZE, callbacks = [early_stopping,model_checkpoint,reduce_lr_loss])
        mediann = keras.models.load_model(save_model_name)
        print('Training is successfull...\n')
        return {'model':mediann,'history': history}


    def train_boost_model(self,model):

        
        """
        Function to use to train a boost model. 
        -----------------------------------------------------------------------------------------
        Input:

        model_pred : The keras object that has been trained
        
        -----------------------------------------------------------------------------------------

        Output:

        model : The trained model

        """
        pred = model['model'].predict(self.input)
        X = self.input #Extracting the X (input)
        target_name = self.target #Storing the target name 
        target = self.output #Replacing the name with the actual array that is our target
        train_set = self.train_set #Extracting training 
        val_set = self.val_set #and validation set 
        target = target-pred #Training on the difference
        if self.lite_model == True:
            new_model = self.mediann_model_lite()
        else:
            new_model = self.mediann_model()
        X_train, Y_train = X[train_set],target[train_set] #Defining target and training set 
        X_val, Y_val = X[val_set],target[val_set] #Defining val target and validation set 
        model_name = 'boost_model'+self.target+'_'+self.experiment_name+'.hdf5'
        print('Training of the boosted machine learning model has started \n')
        #Check the validation loss and stop when the model is overfitting
        #Save the best model as a h5 file
        save_model_name = self.model_path+'/'+model_name
        model_checkpoint = ModelCheckpoint(save_model_name, monitor='val_loss', save_best_only=True)
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=20, factor=0.5, min_lr=1e-5)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        #Store the history of your model
        history = new_model.fit(X_train,Y_train,validation_data = (X_val,Y_val), epochs = self.epochs, batch_size = BATCH_SIZE, callbacks = [early_stopping,model_checkpoint,reduce_lr_loss])
        mediann = keras.models.load_model(save_model_name)
        #mediann = model
        print('Training is successfull...\n')
        return {'history':history,'model':mediann} #Returning the full prediction
    

